import pandas as pd
import pickle
import os
import logging
import openai
import re
import tiktoken
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import numpy as np
import random
# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Model configuration
MODEL = "gpt-4o-mini"  # Default OpenAI model


Path = f"{PROJECT_ROOT}/Dataset/100k/train_test_sets"


trainset_df = pd.read_csv(f'{Path}/u1.base',
                              sep='\t',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              encoding='latin-1')



# API keys and configuration
KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=KEY)


CALL_API = True    # Set to False to skip API calls entirely



# Model pricing in USD per 1M tokens (input, output)
MODEL_PRICING = {
    # OpenAI models (as of March 15, 2025)
    "gpt-4o-mini": (0.60, 2.40),  # $0.30 per 1M input tokens, $1.20 per 1M output tokens
}

# Ollama models are free for local use
# This is just a placeholder for future reference if needed
OLLAMA_MODELS = ["llama3", "mistral", "gemma", "phi3"]

def calculate_tokens(text):
    """Estimate the number of tokens in the text.
    
    Args:
        text: Text to estimate tokens for
        model: Model to use for estimation (different models may have different tokenizers)
        
    Returns:
        int: Estimated number of tokens
    """
    # Simple estimation: ~4 chars per token for English text
    # This is a rough estimate and will vary by model and content
    return max(1, len(text) // 4)


def calculate_cost(input_tokens, output_tokens):
    """Calculate the estimated cost of an API call."""
    input_price, output_price = MODEL_PRICING["gpt-4o-mini"]
    
    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost
    
    return total_cost




# Configure logger
def setup_logger(name='llm_reranker', log_level=logging.INFO):
    """Set up and configure logger with the specified log level.
    
    Args:
        name: Logger name (default: 'llm_reranker')
        log_level: Logging level (default: logging.INFO)
                   Can be set to logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger
    logger.addHandler(ch)
    
    return logger


class LLMReranker:
    """Base class for recommendation reranking."""
    

    RECOMMENDATIONS_FILE = os.path.join(PROJECT_ROOT, 'output/100k/vgcf_recommendations_100k.csv')
    MOVIES_FILE = os.path.join(PROJECT_ROOT, 'Dataset/100k/movies_enriched_dataset_100k.pkl')
    USERS_FILE = os.path.join(PROJECT_ROOT, 'Dataset/100k/updated_users_with_summary_df_100k.csv')

    def __init__(self, log_level=logging.INFO):
        self.logger = setup_logger('llm_reranker', log_level)
        
        self.logger.info(f"Loading data from:\n- Movies: {self.MOVIES_FILE}\n- Users: {self.USERS_FILE}\n- Recommendations: {self.RECOMMENDATIONS_FILE}")
        
        with open(self.MOVIES_FILE, 'rb') as f:
            self.movies_df = pickle.load(f)
        self.logger.info(f"Loaded movies data: {len(self.movies_df)} movies")

        self.users_df = pd.read_csv(self.USERS_FILE)
        self.logger.info(f"Loaded users data: {len(self.users_df)} users")
        
        self.df = pd.read_csv(self.RECOMMENDATIONS_FILE)
        self.logger.info(f"Loaded recommendations data: {len(self.df)} recommendations")
        self.logger.info(f"Unique users in recommendations: {self.df['user_id'].nunique()}")
        self.logger.info(f"Unique movies in recommendations: {self.df['movie_id'].nunique()}")

    def get_top_recommendations_for_all_users(self, n=10):
        """Get top N recommendations for all users based on recommendation_rank."""
        # Group by user_id and get top N recommendations for each user
        top_recommendations = self.df.groupby('user_id', group_keys=False).apply(
            lambda x: x.nsmallest(n, 'recommendation_rank')
        ).reset_index(drop=True)
        
        self.logger.info(f"Generated top {n} recommendations for {top_recommendations['user_id'].nunique()} users")
            
        return top_recommendations

    def get_movie_details(self, movie_id):
        """Get details for a specific movie."""
        movie_data = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if movie_data.empty:
            self.logger.warning(f"Movie ID {movie_id} not found in movies dataset!")
        else:
            self.logger.debug(f"Retrieved details for movie ID {movie_id}: '{movie_data.iloc[0]['title']}'")
        return movie_data
    
    def get_user_details(self, user_id):
        """Get details for a specific user."""
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        if user_data.empty:
            self.logger.warning(f"User ID {user_id} not found in users dataset!")
        else:
            self.logger.debug(f"Retrieved details for user ID {user_id}")
            if 'summary' in user_data.columns:
                self.logger.debug(f"User summary: {user_data['summary'].iloc[0][:100]}...")
        return user_data

    def extract_year_from_release_date(self, release_date_value):
        """Extract year from various release date formats.
        
        Args:
            release_date_value: The release date value to extract year from
            
        Returns:
            str: Extracted year or empty string if not found
        """
        release_year = ""
        
        # Handle different types of release_date values
        if release_date_value is None:
            return release_year
            
        # Convert to string for processing
        release_date_str = str(release_date_value)
        
        # Look for a 4-digit year pattern
        year_match = re.search(r'(19|20)\d{2}', release_date_str)
        if year_match:
            release_year = year_match.group(0)
        else:
            # Just use the first 4 characters if they look like a year
            potential_year = release_date_str[:4]
            if potential_year.isdigit() and 1900 <= int(potential_year) <= 2100:
                release_year = potential_year
                
        return release_year


class LLMRerankerWithOpenAI(LLMReranker):
    """Reranker that uses OpenAI's LLM to rerank recommendations."""
    
    def __init__(self, log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.logger = setup_logger('llm_reranker_openai', log_level)

    def collect_movie_details(self, movie_ids):
        """Collect and format movie details for the given movie IDs.
        
        Args:
            movie_ids: List of movie IDs to collect details for
            
        Returns:
            tuple: (movie_details_list, missing_movies)
                - movie_details_list: List of formatted movie details strings
                - missing_movies: List of movie IDs that were not found
        """
        self.logger.info(f"Collecting details for {len(movie_ids)} movies")
        movie_details_list = []
        missing_movies = []
        
        for movie_id in movie_ids:
            movie_info = self.get_movie_details(movie_id)
            if movie_info.empty:
                missing_movies.append(movie_id)
                continue
                
            # Get the first row as a dictionary
            movie_info_dict = movie_info.iloc[0].to_dict()
            
            # Extract just the year from the release date
            release_year = ""
            if 'release_date' in movie_info_dict:
                release_year = self.extract_year_from_release_date(movie_info_dict['release_date'])
            
            # Clean movie title and ensure it's a string
            title = str(movie_info_dict.get('title', f"Movie {movie_id}"))
            
            # Clean overview and ensure it's a string
            overview = str(movie_info_dict.get('overview', ""))
            
            # Format the movie details with clean information
            movie_details = f"{title} ({release_year}): {overview}"
            movie_details_list.append(movie_details)
            self.logger.debug(f"Added movie {movie_id}: {title} ({release_year})")
        
        if missing_movies:
            self.logger.warning(f"{len(missing_movies)} movies not found: {missing_movies}")
            
        return movie_details_list, missing_movies

    def get_movie_titles(self, movie_ids):
        """Get movie titles for the given movie IDs.
        
        Args:
            movie_ids: List of movie IDs to get titles for
            
        Returns:
            list: List of movie titles
        """
        movie_titles = []
        for mid in movie_ids:
            movie_info = self.get_movie_details(mid)
            if movie_info.empty:
                movie_titles.append("UNKNOWN")
            else:
                movie_info_dict = movie_info.iloc[0].to_dict()
                movie_titles.append(movie_info_dict.get('title', f"Movie {mid}"))
        
        return movie_titles

    def generate_prompt(self, user_summary, movie_details_list):
        prompt = f"""You are an expert movie recommendation system. Your task is to rerank a list of movies based on how well each movie matches a specific user's preferences and viewing history.

USER PROFILE AND PREFERENCES:
{user_summary}

EVALUATION CRITERIA (in order of importance):
1. Direct Preference Match (50% weight):
   - Exact matches with user's explicitly stated preferences
   - Strong alignment with known user interests
   - Match with user's preferred genres and themes

2. Content Similarity (30% weight):
   - Thematic connections to movies user has liked
   - Similar storytelling styles
   - Comparable mood and tone

3. Diversity & Discovery (20% weight):
   - Maintain variety in top recommendations
   - Include some "safe bets" and some novel suggestions
   - Balance between familiar elements and new experiences

RANKING INSTRUCTIONS:
1. Focus heavily on the user's explicit preferences for the top 5 recommendations
2. Ensure the top recommendations are highly relevant and "safe" choices
3. Include more diverse/exploratory options in positions 6-10
4. Consider the user's experience level with different genres
5. For users with limited history, prioritize widely appealing movies within their stated preferences

Do not include any other text in your response except the reranked list of movie titles. Only return the list of movie titles, no other text or explanations.

MOVIES TO RANK:
"""

        for idx, movie_desc in enumerate(movie_details_list, 1):
            prompt += f"{idx}. {movie_desc}\n"

        prompt += "\nRANKED LIST (most to least relevant):"
        
        return prompt
        
    def get_user_recommendations(self, user_id, top_n=10):
        """Get top N recommendations for a specific user.
        
        Args:
            user_id: User ID to get recommendations for
            top_n: Number of top recommendations to retrieve
            
        Returns:
            tuple: (user_summary, user_recommendations, movie_ids)
                - user_summary: User preference summary
                - user_recommendations: DataFrame with user recommendations
                - movie_ids: List of movie IDs to rerank
        """
        # Get user details
        user_details = self.get_user_details(user_id)
        if user_details.empty:
            self.logger.error(f"Cannot rerank recommendations - User ID {user_id} not found!")
            return None, None, None
        
        user_summary = user_details['summary'].iloc[0]
        self.logger.debug(f"User summary available: {len(user_summary) > 0}")

        # Get recommendations for this user
        user_recommendations = self.df[self.df['user_id'] == user_id]
        self.logger.info(f"Found {len(user_recommendations)} total recommendations for user {user_id}")
        
        if len(user_recommendations) == 0:
            self.logger.error(f"No recommendations found for user ID {user_id}!")
            return None, None, None
            
        # Get top N recommendations
        user_recommendations = user_recommendations.nsmallest(top_n, 'recommendation_rank')
        self.logger.info(f"Selected top {len(user_recommendations)} recommendations based on rank")
        self.logger.debug(f"Recommendation ranks: {user_recommendations['recommendation_rank'].tolist()}")
        
        movie_ids = user_recommendations['movie_id'].tolist()
        self.logger.debug(f"Movie IDs to rerank: {movie_ids}")
        
        return user_summary, user_recommendations, movie_ids
    
    def call_llm_api(self, prompt):
        """Call the LLM API with the given prompt."""
        if not CALL_API:
            self.logger.info("Skipping API call (CALL_API=False)")
            return None, None, 0.0, None

        # Calculate input tokens
        input_tokens = calculate_tokens(prompt)
        self.logger.info(f"Prompt length: {len(prompt)} chars, ~{input_tokens} tokens")
        
        self.logger.info("Calling LLM API...")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert movie recommender specializing in cold-start users."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3,
                seed=422,
                top_p=0.15,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            # Get response content
            response_content = response.choices[0].message.content
            self.logger.debug(f"API response received: {len(response_content)} chars")
            
            # Get token usage from response
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate cost
            cost = calculate_cost(
                token_usage["prompt_tokens"], 
                token_usage["completion_tokens"]
            )
            
            self.logger.info(f"Token usage: {token_usage['total_tokens']} tokens " +
                            f"({token_usage['prompt_tokens']} prompt, {token_usage['completion_tokens']} completion)")
            self.logger.info(f"Estimated cost: ${cost:.6f}")
            
            # Process response
            reranked_list = response_content.strip().split("\n")
            self.logger.debug(f"Parsed {len(reranked_list)} items from response")
            self.logger.debug(f"Reranked list: {reranked_list}")
            
            return reranked_list, token_usage, cost, response_content
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return None, None, 0.0, None
    
    def extract_titles_from_response(self, reranked_list):
        """Extract movie titles from the LLM response."""
        try:
            # Clean up the response lines
            cleaned_lines = []
            for line in reranked_list:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Try to extract the title from numbered list format (e.g., "1. Movie Title")
                match = re.match(r'^\s*\d+\.\s*(.*?)(?:\s*\(\d{4}\))?$', line)
                if match:
                    title = match.group(1).strip()
                    # Skip empty titles
                    if not title:
                        continue
                    cleaned_lines.append(title)
                else:
                    # If no match, just add the whole line (after stripping)
                    stripped_line = line.strip()
                    # Skip empty lines
                    if not stripped_line:
                        continue
                    cleaned_lines.append(stripped_line)
            
            self.logger.debug(f"Extracted movie titles: {cleaned_lines}")
            
            # Log the raw response for debugging
            self.logger.debug(f"Raw response lines: {reranked_list}")
            self.logger.debug(f"Cleaned lines: {cleaned_lines}")
            
            return cleaned_lines
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            self.logger.error(f"Raw response: {reranked_list}")
            return None
    
    def apply_new_rankings(self, user_recommendations, reranked_movie_titles, llm_weight=0.7):
        """Apply new rankings with position-dependent weighting."""
        reranked_movies_df = user_recommendations.copy()
        reranked_movies_df['title'] = self.get_movie_titles(reranked_movies_df['movie_id'].tolist())
        
        def normalize_title(title):
            return re.sub(r'[^\w\s]', '', title.lower())
        
        # Filter out any empty titles before creating the map
        valid_titles = [title for title in reranked_movie_titles if title and title.strip()]
        reranked_title_map = {normalize_title(title): idx + 1 for idx, title in enumerate(valid_titles)}
        
        def find_best_match(title):
            normalized = normalize_title(title)
            if normalized in reranked_title_map:
                return reranked_title_map[normalized]
            
            # Try partial match with confidence scoring
            best_match = None
            highest_confidence = 0
            for reranked_title, rank in reranked_title_map.items():
                # Calculate similarity score
                if reranked_title in normalized or normalized in reranked_title:
                    # Check if reranked_title is empty to avoid division by zero
                    if not reranked_title:
                        continue
                    confidence = len(set(reranked_title) & set(normalized)) / len(set(reranked_title))
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = rank
            
            return best_match if best_match is not None else 999
        
        # Get LLM rankings
        reranked_movies_df['llm_rank'] = reranked_movies_df['title'].apply(find_best_match)
        
        # Normalize rankings
        max_orig_rank = reranked_movies_df['recommendation_rank'].max()
        max_llm_rank = reranked_movies_df['llm_rank'].max()
        
        reranked_movies_df['norm_orig_rank'] = reranked_movies_df['recommendation_rank'] / max_orig_rank
        reranked_movies_df['norm_llm_rank'] = reranked_movies_df['llm_rank'] / max_llm_rank
        
        # Position-dependent weighting
        def get_llm_weight(rank):
            if rank <= 5:  # Stronger LLM influence for top 5
                return 0.9
            elif rank <= 10:  # Balanced for next 5
                return 0.7
            else:  # More original ranking weight for the rest
                return 0.5
        
        # Apply position-dependent hybrid scoring
        reranked_movies_df['hybrid_score'] = reranked_movies_df.apply(
            lambda row: (
                get_llm_weight(row['llm_rank']) * row['norm_llm_rank'] +
                (1 - get_llm_weight(row['llm_rank'])) * row['norm_orig_rank']
            ),
            axis=1
        )
        
        # Sort and create new rankings
        reranked_movies_df.sort_values(by='hybrid_score', inplace=True)
        reranked_movies_df['new_rank'] = range(1, len(reranked_movies_df) + 1)
        
        return reranked_movies_df.reset_index(drop=True)

    def rerank_user_recommendations(self, user_id, top_n=10, model=None):
        """Rerank recommendations for a specific user using LLM."""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Reranking recommendations for user ID: {user_id}")
        self.logger.info(f"{'='*50}")
        
        # Use appropriate model if none specified
        if model is None:
            model = MODEL 
        
        # Step 1: Get user recommendations
        user_summary, user_recommendations, movie_ids = self.get_user_recommendations(user_id, top_n)
        if user_summary is None:
            return None, 0.0, None, None

        # Step 2: Collect movie details
        movie_details_list, missing_movies = self.collect_movie_details(movie_ids)
        if len(movie_details_list) == 0:
            self.logger.error("No valid movies to rerank!")
            return None, 0.0, None, None

        # Step 3: Generate prompt
        prompt = self.generate_prompt(user_summary, movie_details_list)

        # Save prompt for debugging
        with open(f'prompt_user_{user_id}.txt', 'w',encoding='utf-8') as f:
            f.write(prompt)
        self.logger.debug(f"Saved prompt to prompt_user_{user_id}.txt")

        # Step 4: Call LLM API
        reranked_list, token_usage, cost, raw_output = self.call_llm_api(prompt)
        if reranked_list is None:
            return None, 0.0, None, None
        
        # Save raw output for display
        with open(f'raw_output_user_{user_id}.txt', 'w') as f:
            f.write(raw_output)
        self.logger.debug(f"Saved raw output to raw_output_user_{user_id}.txt")

        # Step 5: Extract titles from response
        reranked_movie_titles = self.extract_titles_from_response(reranked_list)
        if reranked_movie_titles is None:
            return None, cost, token_usage, raw_output

        # Step 6: Apply new rankings with increased LLM weight (0.85 instead of 0.7)
        reranked_df = self.apply_new_rankings(user_recommendations, reranked_movie_titles, llm_weight=0.85)
        
        return reranked_df, cost, token_usage, raw_output
    
    

def main():
    """Main function to demonstrate usage."""

    # Randomly select 10 users from range 1-943 (MovieLens 100k user range)
    USERS_TO_RERANK = list(range(1,943))
    global MODEL, CALL_API

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Reranker for movie recommendations')
    parser.add_argument('--skip_api', action='store_true', help='Skip API calls entirely')
    parser.add_argument('--top_n', type=int, default=30, help='Number of top recommendations to consider')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    parser.add_argument('--model', type=str, default=MODEL, help='Model to use for reranking')
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    # Set global configuration based on arguments

    
    # Update MODEL if specified
    MODEL = args.model
    
    # Determine which API to use (OpenAI or Ollama)
    if args.skip_api:
        CALL_API = False
    
    
    
    # Set log level
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    log_level = log_level_map.get(args.log_level, logging.INFO)
    
    # Print configuration
    print("\n=== LLM Reranker Configuration ===")
    print(f"Model: {MODEL}")
    print(f"API Calls: {'Enabled' if CALL_API else 'Disabled'}")
    print(f"Log Level: {args.log_level}")
    print(f"Top N: {args.top_n}")
    print("="*35)
    
    # Initialize reranker
    llm_reranker = LLMRerankerWithOpenAI(log_level=log_level)
    
    # Use the specified list of users to process
    users_to_process = USERS_TO_RERANK
    
    print(f"\nProcessing {len(users_to_process)} users: {users_to_process}")
    
    # Initialize list to store all reranked recommendations
    all_reranked_recommendations = []
    total_cost = 0.0
    reranked_users = set()
    
    # Process each user
    for user_id in users_to_process:
        print(f"\nProcessing user {user_id} ({list(users_to_process).index(user_id) + 1}/{len(users_to_process)})")
        
        # Get original recommendations before reranking
        _, original_recommendations, _ = llm_reranker.get_user_recommendations(user_id=user_id, top_n=args.top_n)
        
        # Add titles to original recommendations for better readability
        if original_recommendations is not None:
            original_recommendations['title'] = llm_reranker.get_movie_titles(original_recommendations['movie_id'].tolist())
        
        # Get reranked recommendations
        reranked_recommendations, api_cost, token_usage, raw_output = llm_reranker.rerank_user_recommendations(
            user_id=user_id, top_n=args.top_n, model=args.model
        )
        
        if reranked_recommendations is not None:
            # Add to list of all reranked recommendations
            all_reranked_recommendations.append(reranked_recommendations)
            total_cost += api_cost
            reranked_users.add(user_id)
            
            # Display results for this user
            print(f"\nResults for user {user_id}:")
            print(f"API cost: ${api_cost:.6f}")
            print(f"Token usage: {token_usage['total_tokens'] if token_usage else 'N/A'} tokens")
        else:
            print(f"Failed to rerank recommendations for user {user_id}")
    
    # Get all original recommendations
    original_recommendations_df = llm_reranker.df.copy()
    
    # If we have reranked recommendations, update the original dataframe
    if all_reranked_recommendations:
        reranked_df = pd.concat(all_reranked_recommendations, ignore_index=True)
        
        # Update the original dataframe with reranked data
        for _, row in reranked_df.iterrows():
            mask = (original_recommendations_df['user_id'] == row['user_id']) & (original_recommendations_df['movie_id'] == row['movie_id'])
            original_recommendations_df.loc[mask, 'recommendation_rank'] = row['new_rank']
            original_recommendations_df.loc[mask, 'module_source'] = 'LLM'
    
    # Keep only the specified columns
    output_columns = ['user_id', 'movie_id', 'recommendation_rank', 'module_source']
    final_recommendations_df = original_recommendations_df[output_columns]
    
    # Sort by user_id and recommendation_rank to ensure order matches ranking
    final_recommendations_df = final_recommendations_df.sort_values(['user_id', 'recommendation_rank'])
    
    # Reset index to ensure sequential indices
    final_recommendations_df = final_recommendations_df.reset_index(drop=True)
    
    # Save the complete recommendations to CSV with an unnamed index
    output_file = os.path.join(PROJECT_ROOT, 'output/100k/llm_reranking_recommendations_100k.csv')
    final_recommendations_df.to_csv(output_file, index=True, index_label='')
    print(f"\nSaved recommendations to: {output_file}")
    print(f"Total API cost: ${total_cost:.6f}")
    print(f"Average cost per user: ${total_cost/len(users_to_process):.6f}")
    
    # Print summary statistics
    total_users = len(final_recommendations_df['user_id'].unique())
    reranked_count = len(final_recommendations_df[final_recommendations_df['module_source'] == 'LLM'])
    
    print(f"\nSummary:")
    print(f"Total users in dataset: {total_users}")
    print(f"Total users reranked: {len(reranked_users)}")
    print(f"Users with original rankings: {total_users - len(reranked_users)}")
    print(f"Total recommendations reranked: {reranked_count}")




if __name__ == "__main__":
    main() 