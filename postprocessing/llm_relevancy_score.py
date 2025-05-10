# ---------------------------------------------------------------------------
# LLM‑based Hybrid Batch Re‑ranker –scores (1‑100) instead of pairwise order
# ---------------------------------------------------------------------------
# Date: 2025‑05‑05
# ---------------------------------------------------------------------------
"""A drop‑in replacement for the original ``LLM_Batch_Reranker`` that:

1. Uses the LLM to assign an absolute **relevance score (1‑100)** to each
   candidate instead of a total order.
2. Handles candidates in *over‑lapping* batches to keep prompts small while
   letting every item be judged several times (robustness via averaging).
3. Combines the LLM score with the *original* recommender rank through a
   **hybrid score** (80% LLM, 20% original) before producing the final list.

The public API is unchanged – you can keep calling
```.rerank_all_users(checkpoint_path)```.  The defaults have been adjusted
for a *larger* candidate pool (``pool_k=50``) and for a bigger batch size
(10, overlap 3) so that the model can see more context at once without
hitting token limits on GPT‑4o.
"""

from __future__ import annotations

import json, math, os, time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from postprocessing.batchclass import BatchResult  # unchanged helper dataclass
from postprocessing.LocalLM import LocalLM as llm

load_dotenv()

__all__ = ["LLM_Batch_Reranker"]

# ---------------------------------------------------------------------------
class LLM_Batch_Reranker:
    """Hybrid re‑ranker using batched LLM *scores* instead of pairwise wagers."""

    DEFAULT_MODEL = "gpt-4o-mini"
    SCORE_MIN, SCORE_MAX = 1, 100

    SYSTEM_INSTRUCTIONS = (
        """
        You are a helpful movie expert.  Given five movies and a user profile,
        assign each movie an integer *relevance score* between 1 (terrible)
        and 100 (perfect fit) **for that user**.

        Output **only** valid JSON exactly in the form:
        {
          "scores": {
            "<movie_id>": <int 1‑100>,
            ... x5
          }
        }
        """
    )

    # ------------------------------- initialisation -----------------------
    def __init__(
        self,
        users_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        recs_df: pd.DataFrame,
        *,
        pool_k: int = 50,          # larger pool –gives the LLM more choice
        final_k: int = 20,
        batch_size: int = 10,      # judged together → richer comparative context
        batch_overlap: int = 3,    # each movie seen ≈3× on average
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_llm_retries: int = 5,
        llm_weight: float = 0.8,   # 80% LLM, 20% original
    ) -> None:

        self.users_df = users_df.set_index("user_id")
        self.movies_df = movies_df.set_index("movie_id")
        self.recs_df = recs_df

        self.pool_k = pool_k
        self.final_k = final_k
        self.batch_size = batch_size
        self.batch_overlap = batch_overlap
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("OPENAI_API_KEY_ReRank")
        self.max_llm_retries = max_llm_retries
        self.llm_weight = llm_weight

        # OpenAI + optional local fallback
        self.client = OpenAI(api_key=self.api_key)
        self.local_llm = llm(model="gemma3:4b")

        # bookkeeping
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # cache { (uid, sorted mids tuple ) : {mid: score} }
        self._batch_cache: Dict[Tuple[int, Tuple[int, ...]], Dict[int, int]] = {}

    # ------------------------------- public API ---------------------------
    def rerank_all_users(self, checkpoint_path: Optional[str] = None) -> pd.DataFrame:
        """Iterate over every user; resume safely from `checkpoint_path` if given."""
        processed = set()
        chunks: list[pd.DataFrame] = []

        # resume logic -----------------------------------------------------
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = pd.read_csv(checkpoint_path)
            processed = set(ckpt.user_id.unique())
            chunks.append(ckpt)

        # main loop --------------------------------------------------------
        for uid, group in tqdm(self.recs_df.groupby("user_id"), desc="Reranking users"):
            if uid in processed:
                continue

            # ✨ top‑K candidate pool from downstream model
            pool = (
                group.sort_values("recommendation_rank")
                .head(self.pool_k)["movie_id"].tolist()
            )

            reranked_ids = self._rerank_single_user(uid, pool)[: self.final_k]

            df = pd.DataFrame(
                {
                    "user_id": uid,
                    "movie_id": reranked_ids,
                    "recommendation_rank": range(1, len(reranked_ids) + 1),
                    "module_source": "LLM_Batch_Reranker (hybrid scores)",
                }
            )

            # incremental checkpoint
            if checkpoint_path:
                header = not os.path.exists(checkpoint_path)
                df.to_csv(checkpoint_path, mode="a", header=header, index=False)
                processed.add(uid)

            chunks.append(df)

        return pd.concat(chunks, ignore_index=True)

    # ------------------------------ core logic ---------------------------
    def _rerank_single_user(self, uid: int, movie_ids: List[int]) -> List[int]:
        """Assign LLM scores (w/ batching), fuse with original, then sort."""

        # 1️⃣ split into overlapping batches --------------------------------
        batches: List[List[int]] = []
        step = self.batch_size - self.batch_overlap
        for i in range(0, len(movie_ids) - self.batch_size + 1, step):
            batches.append(movie_ids[i : i + self.batch_size])
        if not batches:                       # safeguard for tiny candidate list
            batches = [movie_ids]

        # 2️⃣ gather scores over all batches --------------------------------
        llm_scores: Dict[int, List[int]] = {mid: [] for mid in movie_ids}
        for mids in batches:
            batch_scores = self._score_batch_with_llm(uid, mids)
            for mid, s in batch_scores.items():
                llm_scores[mid].append(s)

        # average the multiple judgments
        avg_llm_score = {
            mid: (sum(scores) / len(scores) if scores else self.SCORE_MIN)
            for mid, scores in llm_scores.items()
        }

        # 3️⃣ hybrid score --------------------------------------------------
        fused: Dict[int, float] = {}
        # normalise original rank to [SCORE_MIN, SCORE_MAX]
        orig_scale = self.SCORE_MAX - self.SCORE_MIN
        for mid in movie_ids:
            orig_rank = movie_ids.index(mid) + 1  # 1‑based position in *pool* list
            orig_score = (
                self.SCORE_MAX
                - (orig_rank - 1) * orig_scale / (self.pool_k - 1)
            )  # higher rank ⇒ higher score
            fused[mid] = (
                self.llm_weight * avg_llm_score[mid]
                + (1 - self.llm_weight) * orig_score
            )

        # 4️⃣ final sort (desc) ---------------------------------------------
        return sorted(movie_ids, key=lambda m: -fused[m])

    # ----------------------------- LLM helpers ---------------------------
    def _score_batch_with_llm(self, uid: int, mids: List[int]) -> Dict[int, int]:
        """Ask the LLM for *absolute* scores (1‑100) for each movie in *mids*."""
        key = (uid, tuple(sorted(mids)))
        if key in self._batch_cache:
            return self._batch_cache[key]

        prompt = self._build_prompt(uid, mids)

        # try local first (cheaper), then network
        # result = self.local_llm._query_llm_relavancy_score(prompt, mids)
        # result = self._query_llm(prompt, mids)
        result = self._query_llm_o4(prompt, mids)

        # fallback: uniform minimum score
        batch_scores = (
            result.scores if result is not None else {mid: self.SCORE_MIN for mid in mids}
        )
        self._batch_cache[key] = batch_scores
        return batch_scores

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _build_prompt(self, uid: int, mids: List[int]) -> str:
        """Return the user‑visible prompt for *score assignment*."""
        profile = self._get_user_profile(uid)
        movies_txt = "\n\n".join(
            f"Option {i + 1} (movie_id={mid}):\n{self._get_movie_info(mid)}"
            for i, mid in enumerate(mids)
        )
        mids_csv = ", ".join(map(str, mids))
        return f"""
You are a movie recommendation expert.  Analyse the user's tastes and assign a
**relevance score** (integer {self.SCORE_MIN}‑{self.SCORE_MAX}) to *each* of the
FIVE candidate movies below.  Higher = better fit.

### Candidates (IDs: {mids_csv})
{movies_txt}

### User profile
{profile}

### Expected output
Respond **only** with JSON exactly like:
{{
  "scores": {{
    "<movie_id>": "<int {self.SCORE_MIN}-{self.SCORE_MAX}>",
    ...
  }}
}}
"""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _query_llm(self, prompt: str, mids: List[int]):
        """Chat‑completion wrapper returning a ``BatchResult`` with .scores."""
        for attempt in range(self.max_llm_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                    temperature=0.3,
                )
                usage = resp.usage
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens

                content = json.loads(resp.choices[0].message.content)
                raw_scores = content.get("scores", {})

                # ---------- guardrails ----------------------------------
                scores: Dict[int, int] = {}
                for mid in mids:
                    if str(mid) not in raw_scores:
                        raise ValueError(f"Missing score for movie_id {mid}")
                    val = int(raw_scores[str(mid)])
                    if not (self.SCORE_MIN <= val <= self.SCORE_MAX):
                        raise ValueError(f"Score {val} out of range for id {mid}")
                    scores[mid] = val

                return BatchResult(
                    ordered_ids=list(sorted(scores, key=scores.get, reverse=True)),
                    scores=scores,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Bad LLM response (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"⚠️  LLM error: {e}. Retrying…")
                time.sleep(2 ** attempt)

        print("⚠️  LLM failed after all retries –defaulting to min‑scores.")
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _query_llm_o4(self, prompt: str, mids: List[int]):
        """Chat‑completion wrapper returning a ``BatchResult`` with .scores."""
        for attempt in range(self.max_llm_retries):
            try:
                resp = self.client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                usage = resp.usage
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens

                content = json.loads(resp.choices[0].message.content)
                raw_scores = content.get("scores", {})

                # ---------- guardrails ----------------------------------
                scores: Dict[int, int] = {}
                for mid in mids:
                    if str(mid) not in raw_scores:
                        raise ValueError(f"Missing score for movie_id {mid}")
                    val = int(raw_scores[str(mid)])
                    if not (self.SCORE_MIN <= val <= self.SCORE_MAX):
                        raise ValueError(f"Score {val} out of range for id {mid}")
                    scores[mid] = val

                return BatchResult(
                    ordered_ids=list(sorted(scores, key=scores.get, reverse=True)),
                    scores=scores,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Bad LLM response (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"⚠️  LLM error: {e}. Retrying…")
                time.sleep(2 ** attempt)

        print("⚠️  LLM failed after all retries –defaulting to min‑scores.")
        return None

    # ------------------------------------------------------------------
    def _get_user_profile(self, uid: int) -> str:
        return (
            self.users_df.at[uid, "user_profile"]
            if uid in self.users_df.index
            else "No profile available."
        )

    def _get_movie_info(self, mid: int) -> str:
        return (
            self.movies_df.at[mid, "movie_info"]
            if mid in self.movies_df.index
            else "No movie metadata."
        )
