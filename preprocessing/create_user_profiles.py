import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import time
load_dotenv()

# Use your actual API key securely
KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"

# --------------------------------------------------------------
def loading_data(data="100k"):
    user_path = f"../Dataset/{data}/users_with_summary_df.csv"
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

def get_last_n_movies(user_id, train_df, movies_df, n=5):
    """
    Filtering out the last n movies that have received a rating of 1 from the user (negative)
    and the last n movies that have received a rating of 5 from the user (positive).
    Creates integrated text for each set, containing last n movies information from 'movie_info'.
    Returns a dictionary with 'preferred_text' and 'not_interested_text'.
    """

    # filter user rows
    user_rows = train_df[train_df['user_id'] == user_id].sort_values(by='timestamp', ascending=False)

    # last n positive
    positive_movies = user_rows[user_rows['rating'] >= 4].head(n)
    positive_info = []
    for _, row in positive_movies.iterrows():
        movie_id = row['movie_id']
        movie_info_row = movies_df[movies_df['movie_id'] == movie_id]
        if not movie_info_row.empty:
            positive_info.append(movie_info_row.iloc[0]['movie_info'])

    # last n negative
    negative_movies = user_rows[user_rows['rating'] <= 2].head(n)
    negative_info = []
    for _, row in negative_movies.iterrows():
        movie_id = row['movie_id']
        movie_info_row = movies_df[movies_df['movie_id'] == movie_id]
        if not movie_info_row.empty:
            negative_info.append(movie_info_row.iloc[0]['movie_info'])

    preferred_text = "\n".join(positive_info)
    not_interested_text = "\n".join(negative_info)

    return {
        "preferred_text": preferred_text,
        "not_interested_text": not_interested_text
    }

def create_prompt(last_n_movies):
    """
    Generates a refined prompt by combining clear instructions with the results of get_last_n_movies.

    Requirements for the LLM response:
    1) Return a JSON with a single key 'user_profile'.
    2) The contents of 'user_profile' must emphasize user preferences derived from positively rated (rating = 5) movies.
    3) The style and structure of 'user_profile' should be closely aligned with the style found in positive movies (movie_info), facilitating similar vector embeddings.
    4) Factor in negatively rated (rating = 1) movies to identify disliked attributes, but do not copy them into the user_profile style.
    5) The final JSON must be parseable via Python's json.loads.
    6) The user_profile should highlight relevant genres, themes, or features learned from the positive samples.
    """

    # Provide more structured instructions for the user profile
    instruction_part = """
You are an expert in detecting user's movie preferences. You have a set of positively rated (rating 4 or above) and negatively rated (rating 2 or less) movies.
Analyze the textual descriptions of positive and negative ratings to capture user's cinematic preferences.
    
Create a well-structured, user-friendly **markdown** profile with the following sections and fill in the details and enrich the profile with based on your analysis.

Expected output structure:


## Overview
Brief description or summary of the user's preferences. (maximum 1 paragraph)

## Attributes
- **Genre**: List all preferred genres. (You are only allowed to use vocabulary that is present in movie profiles)
- **Tags**: List of all descriptive preferred tags or keywords. (You are only allowed to use vocabulary that is present in movie profiles)

## Description
Detailed textual description, unique attributes of users tastes. (maximum 1 short paragraph)

## Dislikes:
List of specific keywords or tags that user doesn't like based on their ratings. (You are only allowed to use vocabulary that is present in movie profiles)


Format your final response strictly as valid markdown following above structure.
    
"""

    prompt = f"""
{instruction_part}

User watch history:

### Positively rated Movies Info:

{last_n_movies['preferred_text']}


### Negatively rated Movies Info:

{last_n_movies['not_interested_text']}

"""

    return prompt

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
                return output, tokens


            except json.JSONDecodeError:
                print(f"Invalid JSON from LLM on attempt {attempt + 1}. Retrying...")

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

    data = "100k"

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

    for user_id in tqdm(user_ids, desc="Processing users and generating summary profile"):
        # skip if there's already a user_profile in users_df
        current_profile = users_df.loc[users_df['user_id'] == user_id, 'user_profile'].values[0]
        if current_profile is not None and pd.notna(current_profile):
            # if user_profile is already filled, skip
            continue

        # extract last_n_movies using get_last_n_movies()
        last_n_movies = get_last_n_movies(user_id, train_df, movies_df, n=10)

        # creating prompt for the user
        prompt = create_prompt(last_n_movies)

        # generating user_profile
        user_profile, tokens = get_llm_response(prompt, instructions)

        # update token usage
        if tokens and 'total_tokens' in tokens:
            temp_token_counter += tokens['total_tokens']

        # check if 60 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            # if within the same minute, check token usage
            if temp_token_counter > 195000:
                # sleep 10 seconds to respect rate limit
                time.sleep(10)
        else:
            # reset counters if more than 60 seconds have passed
            start_time = time.time()
            temp_token_counter = 0

        # save user_profile into users_df
        users_df.loc[users_df['user_id'] == user_id, 'user_profile'] = user_profile
        users_df.to_csv(f"users_structured_{data}.csv", index=False)

    # at the end, you might want to save users_df back to csv
    users_df.to_csv(f"users_structured_{data}.csv", index=False)


