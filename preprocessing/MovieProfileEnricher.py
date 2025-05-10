from datetime import time
from local_llm.LocalLM import LocalLM
from openai import OpenAI
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# Use your actual API key securely
API_KEY = os.getenv('OPENAI_API_KEY')


class MovieProfileEnricher:
    def __init__(self, openai_api_key: str):
        """
        Initializes the MovieProfileEnricher with the OpenAI API key and model name.

        :param openai_api_key: Your OpenAI API key.
        :param model_name: The OpenAI model to use, e.g., 'gpt-3.5-turbo' or 'gpt-4'.
        """
        self.api_key = openai_api_key
        self.model = "gpt-4o-mini"
        openai.api_key = self.api_key

        self.instructions = "You are a Movie expert who knows everything about movies"

    def get_prompt(self, movie_info: str) -> str:
        """
        Sends the movie_info to the OpenAI model with a carefully crafted prompt
        and returns the structured markdown output as a string.
        """

        prompt = f"""
You are a helpful movie expert. You have the following information about a movie:

{movie_info}

Create a well-structured, user-friendly **markdown** profile with the following sections and use your knowledge of movies to fill in the details and enrich the profile with additional informatiom.
if you don't have additional information, leave that section empty.
        
Expected output structure:

# Movie Name

## Overview
Brief description or summary of the movie.

## Attributes
- **Genre**: List all relevant genres.
- **Tags**: List of all descriptive tags or keywords relevant to the movie.

## Description
Detailed textual description, plot summary (movie) and unique attributes. (maximum 1 short paragraph)


## Dislikes:
List of specific keywords that are NOT present in the movie without external explanation (Opposite of main the genres and tags)


Format your final response strictly as valid markdown following above structure.
        """

        return prompt

    def enrich_dataframe(self, content_df: pd.DataFrame, interim_csv_path: str) -> pd.DataFrame:
        """
        Takes a DataFrame containing 'movie_info' (and presumably 'movie_id'),
        calls 'generate_movie_profile' for each row, and returns a new dataframe
        with an additional column 'structured_info'.

        Uses tqdm for a progress bar and writes the updated DataFrame to CSV after
        each LLM response to safeguard against interruptions.
        """
        # Ensure the column exists so we can update it row-by-row
        # if "structured_info" not in content_df.columns:
        #     content_df["structured_info"] = None

        llm = LocalLM(model="gemma3:12b")

        # Iterate through rows with progress bar
        for idx, row in tqdm(content_df.iterrows(), total=len(content_df), desc="Enriching rows"):
            if pd.isna(row["structured_info"]) or row["structured_info"] == "":
                movie_info = row["movie_info"]
                prompt = self.get_prompt(movie_info)
                enriched_text = self.get_llm_response(prompt)
                # enriched_text = llm.get_llm_response(prompt=prompt)
                # Update the DataFrame with the new info
                content_df.at[idx, "structured_info"] = enriched_text
                # Write progress to CSV after each iteration
                self.export_to_csv(content_df, interim_csv_path)

        return content_df

    def export_to_csv(self, content_df: pd.DataFrame, output_path: str):
        """
        Exports the DataFrame to a CSV file.
        """
        content_df.to_csv(output_path, index=False)

    def get_llm_response(self, prompt, max_retries=3):
        """
        sending the prompt to the LLM and get back the response
        """

        openai.api_key = self.api_key
        instructions = self.instructions

        client = OpenAI(api_key=self.api_key)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=700,
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
                    if isinstance(output, str):
                        return output

                except:
                    print(f"Invalid output on attempt {attempt + 1}. Retrying...")

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


    def merge_structured_info(self, df_with_info, df_without_info, output_csv_path=None):
        """
        Merges the "structured_info" column from df_with_info into df_without_info based on matching "title".

        Parameters:
            df_with_info (pd.DataFrame): DataFrame that contains both 'title' and 'structured_info'.
            df_without_info (pd.DataFrame): DataFrame that contains 'title' but lacks 'structured_info'.
            output_csv_path (str, optional): If provided, the resulting DataFrame is saved to this path.

        Returns:
            pd.DataFrame: The updated DataFrame with the 'structured_info' column filled where possible.
        """
        df_with_info.columns = df_with_info.columns.str.lower()
        df_without_info.columns = df_without_info.columns.str.lower()

        # Normalize the 'title' columns in both DataFrames: convert to string, strip whitespace, and lowercase.
        # df_with_info['title'] = df_with_info['title'].astype(str).str.strip().str.lower()
        # df_without_info['title'] = df_without_info['title'].astype(str).str.strip().str.lower()

        # Create a temporary DataFrame with unique titles and their corresponding structured_info.
        # This avoids any ambiguity if there are duplicate titles in df_with_info.
        unique_info = df_with_info.drop_duplicates(subset='title')[['title', 'structured_info']]

        # Merge using a left join so that all rows from df_without_info are preserved.
        df_merged = pd.merge(df_without_info, unique_info, on='title', how='left')

        # Optionally, save the updated DataFrame to a CSV file.
        if output_csv_path:
            df_merged.to_csv(output_csv_path, index=False)

        return df_merged


if __name__ == "__main__":
    # 1. Load your DataFrame (containing 'movie_info' and 'movie_id')
    data = "1m"
    content_df = pd.read_csv(f"movies/movies_enriched_dataset_{data}.csv")
    content_df_100k = pd.read_csv(f"movies/movies_structured_100K.csv")

    # 2. Drop rows where 'movie_info' is null
    content_df = content_df.dropna(subset=["movie_info"])
    # content_df["structured_info"] = None

    # 3. Initialize your Enricher
    enricher = MovieProfileEnricher(openai_api_key=API_KEY)
    content_df = enricher.merge_structured_info(df_with_info=content_df_100k, df_without_info=content_df, output_csv_path=f"movies_structured_not_complete_{data}.csv")

    # 4. Enrich the DataFrame
    enriched_df = enricher.enrich_dataframe(content_df, interim_csv_path=f"movies_structured_{data}.csv")

    print("CSV with enriched movie profiles created at enriched_movies.csv")
