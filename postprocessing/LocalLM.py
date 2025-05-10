import json
import time
from typing import Dict, List, Optional, Tuple
import torch
import ollama
from postprocessing.batchclass import BatchResult


class LocalLM:

    def __init__(self, model):
        # Initialize the Ollama client
        self.client = ollama.Client()
        self.model = model

    def preprocess_and_parse_json(self, response):
        # Remove any leading/trailing whitespace and newlines
        if response.startswith('```json') and response.endswith('```'):
            cleaned_response = response[len('```json'):-len('```')].strip()

        # Parse the cleaned response into a JSON object
        try:
            json_object = json.loads(cleaned_response)
            return json_object
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None


    def _query_llm(self, prompt, mids: List[int], expect: int, max_retries=10):
        """
        Send the prompt to the LLM and get back the response.
        Includes handling for GPU memory issues by clearing cache and waiting before retry.
        """
        for attempt in range(max_retries):
            try:
                # Try generating the response
                response = self.client.generate(model=self.model, prompt=prompt)
            except Exception as e:
                # This catches errors like the connection being forcibly closed
                print(f"Error on attempt {attempt + 1}: {e}.")
                try:
                    # Clear GPU cache if you're using PyTorch; this may help free up memory
                    torch.cuda.empty_cache()
                    print("Cleared GPU cache.")
                except Exception as cache_err:
                    print("Failed to clear GPU cache:", cache_err)
                # Wait a bit before retrying to allow memory to recover
                time.sleep(2)
                continue

            try:

                try:
                    content = self.preprocess_and_parse_json(response.response)
                    ordered_ids = [int(x) for x in content["ordered_movie_ids"]]

                    # Ensure all returned IDs are in the original mids list
                    if len(ordered_ids) != expect or any(mid not in mids for mid in ordered_ids):
                        raise ValueError("Invalid movie IDs in response")

                    if content is None:
                        continue

                    return BatchResult(
                        ordered_ids=ordered_ids,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    )

                except json.JSONDecodeError:
                    print(f"Invalid JSON from LLM on attempt {attempt + 1}. Retrying...")
            except Exception as parse_error:
                print("Error processing output:", parse_error)

        print("Max retries exceeded. Returning empty response.")
        return None


    def _query_llm_relavancy_score(self, prompt: str, mids: List[int], max_retries=5, SCORE_MIN=1, SCORE_MAX=100):
        """Chat‑completion wrapper returning a ``BatchResult`` with .scores."""
        for attempt in range(max_retries):
            try:
                # Try generating the response
                response = self.client.generate(model=self.model, prompt=prompt)
            except Exception as e:
                # This catches errors like the connection being forcibly closed
                print(f"Error on attempt {attempt + 1}: {e}.")
                try:
                    # Clear GPU cache if you're using PyTorch; this may help free up memory
                    torch.cuda.empty_cache()
                    print("Cleared GPU cache.")
                except Exception as cache_err:
                    print("Failed to clear GPU cache:", cache_err)
                # Wait a bit before retrying to allow memory to recover
                time.sleep(2)
                continue

            try:

                content = self.preprocess_and_parse_json(response.response)
                raw_scores = content.get("scores", {})

                # ---------- guardrails ----------------------------------
                scores: Dict[int, int] = {}
                for mid in mids:
                    if str(mid) not in raw_scores:
                        raise ValueError(f"Missing score for movie_id {mid}")
                    val = int(raw_scores[str(mid)])
                    if not (SCORE_MIN <= val <= SCORE_MAX):
                        raise ValueError(f"Score {val} out of range for id {mid}")
                    scores[mid] = val

                return BatchResult(
                    ordered_ids=list(sorted(scores, key=scores.get, reverse=True)),
                    scores=scores,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ Bad LLM response (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"⚠️  LLM error: {e}. Retrying…")
                time.sleep(2 ** attempt)

        print("⚠️  LLM failed after all retries –defaulting to min‑scores.")
        return None


### o4 for later

    # def _query_llm_o4(self, prompt: str, mids: List[int], expect: int):
    #     """Call the chat‑completion endpoint with exponential‑backoff retries."""
    #     for attempt in range(self.max_llm_retries):
    #
    #         try:
    #             resp = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[
    #                     {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
    #                     {"role": "user", "content": prompt},
    #                 ],
    #                 response_format={"type": "json_object"},
    #             )
    #             usage = resp.usage
    #             self.total_prompt_tokens += usage.prompt_tokens
    #             self.total_completion_tokens += usage.completion_tokens
    #
    #             content = json.loads(resp.choices[0].message.content)
    #             ordered_ids = [int(x) for x in content["ordered_movie_ids"]]
    #
    #             # Ensure all returned IDs are in the original mids list
    #             if len(ordered_ids) != expect or any(mid not in mids for mid in ordered_ids):
    #                 raise ValueError("Invalid movie IDs in response")
    #
    #             return BatchResult(
    #                 ordered_ids=ordered_ids,
    #                 prompt_tokens=usage.prompt_tokens,
    #                 completion_tokens=usage.completion_tokens,
    #                 total_tokens=usage.total_tokens,
    #             )
    #
    #         except (json.JSONDecodeError, KeyError, ValueError) as e:
    #             print(f"Bad LLM response (attempt {attempt + 1}): {e}")
    #         except Exception as e:
    #             print(f"LLM error: {e}. Retrying…")
    #             time.sleep(2 ** attempt)
    #
    #     print("⚠️  LLM failed after retries – using fallback order.")
    #     return None