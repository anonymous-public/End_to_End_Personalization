import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import time
import re
load_dotenv()

# Use your actual API key securely
KEY = os.getenv('OPENAI_API_KEY')
# MODEL = "gpt-4o-mini"
MODEL = "gpt-4.1-mini"



# ---------------------------------------------------------------------------
# Prompt builders ---------------------------------------------------------
INITIAL_INSTRUCTION = """
You are an expert at detecting user's movie preferences. Using the **FIRST 5 positively rated movies** below, CREATE 
a well-structured, user-friendly **markdown** profile with the following sections and fill in the details and enrich the profile 
using only vocabulary from the movie descriptions.

     
Expected output structure:

## Overview
Brief description or summary of the user profile. (maximum 2-3 sentences)

## Attributes
- **Genre**: List of preferred genres (You are only allowed to use vocabulary that is present in movie profiles) 
- **Tags**: List of preferred general descriptive tags. (You are only allowed to use vocabulary that is present in movie profiles)

## Description
Brief textual description and unique attributes. (maximum 1 short paragraph)
(You are only allowed to use description and attribute vocabulary that is present in movie profiles)


## Dislikes:
List of specific keywords and tags that user does not prefer.
(You are only allowed to use description and attribute vocabulary that is present in movie profiles)

Format your final response strictly as valid markdown following above structure.

"""

REFINE_INSTRUCTION = """
You already have a markdown user profile. Using the **ADDITIONAL EVIDENCE**
provided (either more liked movies or disliked movies), UPDATE the existing
profile *in place*:

* keep the same markdown structure, sections and order,
* Adjust Overview, Genre, Tags, description and Dislikes based on the new evidence, if needed.
* do **NOT** add extra sections,
* return the **entire updated markdown profile only**.


Expected output structure:

## Overview
Brief description or summary of the user profile. (maximum 2-3 sentences)

## Attributes
- **Genre**: List of preferred genres (You are only allowed to use vocabulary that is present in movie profiles) 
- **Tags**: List of preferred general descriptive tags. (You are only allowed to use vocabulary that is present in movie profiles)

## Description
Brief textual description and unique attributes. (maximum 1 short paragraph)
(You are only allowed to use description and attribute vocabulary that is present in movie profiles)


## Dislikes:
List of specific keywords and tags that user does not prefer.
(You are only allowed to use description and attribute vocabulary that is present in movie profiles)

Format your final response strictly as valid markdown following above structure.
"""





# --------------------------------------------------------------
def loading_data(data="100k"):
    user_path = f"../Dataset/{data}/users_with_summary_df_{data}.csv"
    movie_path = f"movies/movies_structured_{data}.csv"
    trainset_path = f"../Dataset/{data}/train_test_sets/u1.base"

    users_df = pd.read_csv(user_path)
    users_df = users_df[['user_id']]
    # In case 'user_profile' column does not exist, create it
    if 'user_profile' not in users_df.columns:
        users_df['user_profile'] = None

    movies_df = pd.read_csv(movie_path)
    movies_df = movies_df[['movie_id', 'movie_info']]
    train_df = pd.read_csv(trainset_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')

    return users_df, movies_df, train_df

def _fetch_movie_infos(movie_ids: list[int], movies_df: pd.DataFrame) -> list[str]:
    """Return the ordered list of movie_info strings for the supplied ids."""
    infos = []
    for mid in movie_ids:
        info_row = movies_df.loc[movies_df["movie_id"] == mid, "movie_info"]
        if not info_row.empty:
            infos.append(info_row.iloc[0])
    return infos


def _strip_code_fences(text: str) -> str:
    """
    Remove a single leading/trailing triple‑backtick fence (with or without
    language tag) from `text`. Returns the cleaned string.
    """
    fence_pattern = r"^\s*```(?:\w+)?\s*(.*?)\s*```$"  # DOTALL match
    m = re.match(fence_pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def get_user_interactions(
    user_id: int,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    pos_n: int = 10,
    neg_n: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Return two *ordered* lists (latest‑first) with the last `pos_n` positive and
    last `neg_n` negative movie‑info strings for `user_id`.
    """
    rows = train_df[train_df["user_id"] == user_id].sort_values(
        "timestamp", ascending=False
    )
    pos_ids = rows.loc[rows["rating"] >= 4, "movie_id"].head(pos_n).tolist()
    neg_ids = rows.loc[rows["rating"] <= 2, "movie_id"].head(neg_n).tolist()
    return _fetch_movie_infos(pos_ids, movies_df), _fetch_movie_infos(neg_ids, movies_df)


def _in_batches(lst: list[str], size: int = 5) -> list[list[str]]:
    """
    Split `lst` into consecutive chunks of *up to* `size` items.
    The final batch is returned even if it is shorter than `size`.
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]

def build_initial_prompt(pos_movies: list[str]) -> str:
    pos_text = "\n".join(pos_movies)
    return f"{INITIAL_INSTRUCTION}\n### Positively rated Movies:\n{pos_text}"


def build_refine_prompt(
    existing_profile: str,
    new_positives: list[str] | None = None,
    new_negatives: list[str] | None = None,
) -> str:
    parts = [f"## EXISTING PROFILE\n{existing_profile}"]
    if new_positives:
        parts.append(f"## NEW POSITIVE MOVIES\n" + "\n".join(new_positives))
    if new_negatives:
        parts.append(f"## NEW NEGATIVE MOVIES\n" + "\n".join(new_negatives))
    return "\n\n".join(parts)


def get_llm_response(prompt, instructions, max_retries=3):
    """
    sending the prompt to the LLM and get back the response
    """

    openai.api_key = KEY
    client = OpenAI(api_key=KEY)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ],
                # response_format={"type": "json_object"},
                max_tokens=500,
                n=1,
                temperature=0.7
            )

            tokens = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            try:
                output = response.choices[0].message.content
                output = _strip_code_fences(output)
                return output, tokens


            except json.JSONDecodeError:
                print(f"Invalid response from LLM on attempt {attempt + 1}. Retrying...")

        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.

        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            backoff_time = (2 ** attempt)
            time.sleep(backoff_time)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

    print("Max retries exceeded. Returning empty response.")
    return [], {}
# --------------------------------------------------------------

if __name__ == "__main__":

    data = "1m"

    # loading data
    users_df, movies_df, train_df = loading_data(data=data)
    # users_df = pd.read_csv(f"updated_users_with_summary_df_{data}.csv")

    # set LLM initial instruction
    instructions = """You are a Movie expert who knows everything about movies and can create a perfect detailed user profiles based on their watched history"""

    # Filtering out invalid movie_ids, make sure we have movie_info for every movie in the test set
    user_ids = train_df['user_id'].unique()

    temp_token_counter = 0
    previous_token_total = 0

    # track time in seconds
    start_time = time.time()

    # create a new column as user_profile in users_df which we can populate it during the process
    # (already handled in loading_data if not present)

    for user_id in tqdm(user_ids, desc="Generating interactive profiles"):
        current_profile = users_df.loc[users_df["user_id"] == user_id, "user_profile"].values[0]
        if pd.notna(current_profile):
            continue  # already done

        # Get interactions & split into batches
        pos_movies, neg_movies = get_user_interactions(user_id, train_df, movies_df)

        if not pos_movies:  # ⬅️ NEW: skip only when *zero* positives
            continue

        pos_batches = _in_batches(pos_movies, 5)  # now always ≥1 batch
        neg_batches = _in_batches(neg_movies, 5)

        # Initial profile from first 5 positives
        prompt = build_initial_prompt(pos_batches[0])
        profile_md, tokens = get_llm_response(prompt, INITIAL_INSTRUCTION)
        if not profile_md:
            continue  # safeguard

        temp_token_counter += tokens.get("total_tokens", 0)

        # Refine with remaining positives (at most one more batch)
        for batch in pos_batches[1:]:
            prompt = build_refine_prompt(profile_md, new_positives=batch)
            profile_md, tokens = get_llm_response(prompt, REFINE_INSTRUCTION)
            temp_token_counter += tokens.get("total_tokens", 0)

        # Refine twice with negatives (2 × 5)
        for batch in neg_batches:
            prompt = build_refine_prompt(profile_md, new_negatives=batch)
            profile_md, tokens = get_llm_response(prompt, REFINE_INSTRUCTION)
            temp_token_counter += tokens.get("total_tokens", 0)

        # Save & persist
        users_df.loc[users_df["user_id"] == user_id, "user_profile"] = profile_md
        users_df.to_csv(f"users_structured_{data}.csv", index=False)

        # rate‑limit handling (unchanged)
        elapsed_time = time.time() - start_time
        if elapsed_time < 60 and temp_token_counter > 195000:
            time.sleep(10)
        elif elapsed_time >= 60:
            start_time = time.time()
            temp_token_counter = 0


