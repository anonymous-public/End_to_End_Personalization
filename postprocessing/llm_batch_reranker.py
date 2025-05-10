# ---------------------------------------------------------------------------
# Imports and Setup  (unchanged – shown for context)
# ---------------------------------------------------------------------------
import os, json, time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from postprocessing.batchclass import BatchResult
from postprocessing.LocalLM import LocalLM as llm
from trueskill import Rating, rate_1vs1
# from LocalLM import LocalLM
load_dotenv()



# ---------------------------------------------------------------------------
class LLM_Batch_Reranker:
    """
    Reranks recommendations with *batched* (size=5) LLM calls.
    Steps
    -----
    1.  Take the top‑30 produced by the downstream recommender.
    2.  Split them into *overlapping* batches of 5:
            [0‑4], [3‑7], [6‑10], …, [24‑28]   (overlap = 2 items)
        – 7 batches, 30 distinct ids, ≈ 35 unique comparisons.
    3.  Ask the LLM to ORDER the 5 ids in each batch.
    4.  Aggregate the partial orders with a Borda‑count, producing
        a single global ranking; keep the top‑20.
    5.  Checkpoint continuously to allow safe restarts.
    """
    DEFAULT_MODEL = "gpt-4o-mini"
    # DEFAULT_MODEL = "o4-mini"
    SYSTEM_INSTRUCTIONS = (
        "You are a helpful movie expert.  Using the user profile, "
        "rank the FIVE given movies from *best* to *worst* for that user.  "
        "Be concise and strictly follow the response format."
    )

    # ------------------------------- initialisation -----------------------
    def __init__(
            self,
            users_df: pd.DataFrame,
            movies_df: pd.DataFrame,
            recs_df: pd.DataFrame,
            pool_k: int = 30,  # size of the candidate pool
            final_k: int = 20,  # size of the final list we keep
            batch_size: int = 5,
            batch_overlap: int = 2,  # items shared between consecutive batches
            model: str | None = None,
            api_key: str | None = None,
            max_llm_retries: int = 5,
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

        self.client = OpenAI(api_key=self.api_key)
        self._batch_cache: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.local_llm = llm(model="gemma3:12b")

    # ------------------------------- public API ---------------------------
    def rerank_all_users(self, checkpoint_path: Optional[str] = None) -> pd.DataFrame:
        """Iterate through every user; resume from a checkpoint if present."""
        processed = set()
        chunks: list[pd.DataFrame] = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = pd.read_csv(checkpoint_path)
            processed = set(ckpt.user_id.unique())
            chunks.append(ckpt)

        for uid, group in tqdm(self.recs_df.groupby("user_id"), desc="Reranking users"):
            if uid in processed:
                continue

            pool = (
                group.sort_values("recommendation_rank")  # downstream rank
                .head(self.pool_k)["movie_id"]
                .tolist()
            )
            reranked = self._rerank_single_user(uid, pool)[: self.final_k]

            df = pd.DataFrame(
                {
                    "user_id": uid,
                    "movie_id": reranked,
                    "recommendation_rank": range(1, len(reranked) + 1),
                    "module_source": "LLM_Batch_Reranker",
                }
            )

            # append to checkpoint
            if checkpoint_path:
                header = not os.path.exists(checkpoint_path)
                df.to_csv(checkpoint_path, mode="a", header=header, index=False)
                processed.add(uid)

            chunks.append(df)

        return pd.concat(chunks, ignore_index=True)

    # ------------------------------ core logic ---------------------------
    # def _rerank_single_user(self, user_id: int, movie_ids: List[int]) -> List[int]:
    #     """Rank a single user's candidate list via overlapping 5‑sized batches."""
    #     # 1️⃣ build overlapping batches
    #     batches: List[List[int]] = []
    #     step = self.batch_size - self.batch_overlap
    #     for i in range(0, len(movie_ids) - self.batch_size + 1, step):
    #         batches.append(movie_ids[i: i + self.batch_size])
    #
    #     # 2️⃣ ask the LLM to order each batch
    #     borda_scores: Dict[int, int] = {mid: 0 for mid in movie_ids}
    #     for batch in batches:
    #         ordered = self._rank_batch_with_llm(user_id, batch)
    #         # Borda count: pos 0 gets 4 points, pos 4 gets 0
    #         for score, mid in enumerate(reversed(ordered)):
    #             borda_scores[mid] += score
    #
    #     # 3️⃣ global sort by descending score (higher → better)
    #     return sorted(movie_ids, key=lambda m: -borda_scores[m])

    def _rerank_single_user(self, user_id: int, movie_ids: List[int]) -> List[int]:

        """Rank a single user's candidate list via overlapping 5‑sized batches."""
        # 1️⃣ build overlapping batches
        batches: List[List[int]] = []
        step = self.batch_size - self.batch_overlap
        for i in range(0, len(movie_ids) - self.batch_size + 1, step):
            batches.append(movie_ids[i: i + self.batch_size])

        ratings = {mid: Rating() for mid in movie_ids}
        for batch in batches:
            ordered = self._rank_batch_with_llm(user_id, batch)
            # update pair‑wise: winner beats everyone below it
            for i, winner in enumerate(ordered):
                for loser in ordered[i + 1:]:
                    ratings[winner], ratings[loser] = rate_1vs1(
                        ratings[winner], ratings[loser])

        return sorted(movie_ids, key=lambda m: -ratings[m].mu)

    # ----------------------------- LLM helpers ---------------------------
    def _build_prompt(self, uid: int, mids: List[int]) -> str:
        profile = self._get_user_profile(uid)
        movies_txt = "\n\n".join(
            f"Option {i + 1} (movie_id={mid}):\n{self._get_movie_info(mid)}"
            for i, mid in enumerate(mids)
        )

        mids_csv = ", ".join(map(str, mids))
        return f"""

You are a movie recommendation expert. Analyze the user's preferences and rank movies considering alignments in:
- Overview
- Attributes like **Genre** **Tags**
- Description
- and Dislikes


Return ONLY valid JSON with ordered_movie_ids from BEST to WORST.

### <Task>

You are given **FIVE** candidate movies (IDs: {mids_csv}).
Rank them from *most* to *least* relevant for the user.

{movies_txt}

### User profile
Below is a user profile and everything we know about their movie preferences:

{profile}

Respond **only** with valid JSON of exactly the form:
{{
  "ordered_movie_ids": [{mids_csv}]   // BEST ➜ WORST
}}
"""

    # ----------------------------- LLM helpers ---------------------------
    def _rank_batch_with_llm(self, uid: int, mids: List[int]) -> List[int]:
        """Return a *total order* of the 5 ids (best ➜ worst)."""
        key = (uid, tuple(sorted(mids)))
        if key in self._batch_cache:
            return self._batch_cache[key]

        prompt = self._build_prompt(uid, mids)
        # result = self._query_llm(prompt, mids, expect=len(mids))  # Pass mids here
        # result = self._query_llm_o4(prompt, mids, expect=len(mids))
        result = self.local_llm._query_llm(prompt, mids, expect=len(mids))

        if result is None:
            # fallback: keep original downstream order for this batch
            ordered = mids
        else:
            ordered = result.ordered_ids

        self._batch_cache[key] = ordered
        return ordered

    def _query_llm(self, prompt: str, mids: List[int], expect: int):
        """Call the chat‑completion endpoint with exponential‑backoff retries."""
        for attempt in range(self.max_llm_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=100,
                    temperature=0.5,
                )
                usage = resp.usage
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens

                content = json.loads(resp.choices[0].message.content)
                ordered_ids = [int(x) for x in content["ordered_movie_ids"]]

                # Ensure all returned IDs are in the original mids list
                if len(ordered_ids) != expect or any(mid not in mids for mid in ordered_ids):
                    raise ValueError("Invalid movie IDs in response")

                return BatchResult(
                    ordered_ids=ordered_ids,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Bad LLM response (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"LLM error: {e}. Retrying…")
                time.sleep(2 ** attempt)

        print("⚠️  LLM failed after retries – using fallback order.")
        return None


    # ............................... utilities ..........................
    def _get_user_profile(self, uid: int) -> str:
        return self.users_df.at[uid, "user_profile"] if uid in self.users_df.index else "No profile"

    def _get_movie_info(self, mid: int) -> str:
        return self.movies_df.at[mid, "movie_info"] if mid in self.movies_df.index else "No info"
