import pandas as pd
# from llm_batch_reranker import LLM_Batch_Reranker
from llm_relevancy_score import LLM_Batch_Reranker



# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def loading_data(data, recommender_method):
    user_path = f"../Dataset/preprocessed/users_structured_{data}.csv"
    movie_path = f"../Dataset/preprocessed/movies_structured_{data}.csv"
    recommendation_path = f"../output/{data}/{recommender_method}_recommendations_{data}.csv"

    users_df = pd.read_csv(user_path)
    users_df = users_df[['user_id', 'user_profile']]

    movies_df = pd.read_csv(movie_path)
    movies_df = movies_df[['movie_id', 'movie_info']]

    recommendation_df = pd.read_csv(recommendation_path)
    # module_source is a text representing the recommender system algorithm
    recommendation_df = recommendation_df[['user_id','movie_id', 'recommendation_rank', 'module_source']]

    return users_df, movies_df, recommendation_df



# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = "100k"
    method = "vgcf_new"
    users_df, movies_df, recs_df = loading_data(data=data, recommender_method=method)

    # Filter rows where 'summary' and 'movie_info' are not null
    users_df = users_df.dropna(subset=["user_profile"])
    movies_df = movies_df.dropna(subset=["movie_info"])

    # record original sizes
    orig_user_count = users_df.shape[0]
    orig_movie_count = movies_df.shape[0]

    # filter down to only those in recs_df
    users_df = users_df[users_df['user_id'].isin(recs_df['user_id'])]
    movies_df = movies_df[movies_df['movie_id'].isin(recs_df['movie_id'])]

    # compute how many were dropped
    dropped_users = orig_user_count - users_df.shape[0]
    dropped_movies = orig_movie_count - movies_df.shape[0]

    print(f"Eliminated {dropped_users} users (from {orig_user_count} to {users_df.shape[0]})")
    print(f"Eliminated {dropped_movies} movies (from {orig_movie_count} to {movies_df.shape[0]})")

    # # --------- sample users --------------------
    # # 1. Sample 5 unique users and keep as a DataFrame
    # sample_users = (
    #     recs_df[['user_id']]
    #     .drop_duplicates()
    #     .sample(n=5)
    #     .reset_index(drop=True)
    # )
    #
    # # 2. Filter recommendations for just the sampled users
    # sampled_recs = recs_df[
    #     recs_df['user_id'].isin(sample_users['user_id'])
    # ].reset_index(drop=True)
    #
    # # 3. Keep only the desired columns
    # sampled_recs = sampled_recs[["user_id", "movie_id", "recommendation_rank", "module_source"]]

    # --------- sample users --------------------

    # file_name = f"{method}_recommendations_LLM_Heap_k20_batch.csv"
    file_name = f"{method}_recommendations_relevancy_score_k20_o4_{data}.csv"

    reranker = LLM_Batch_Reranker(users_df, movies_df, recs_df, pool_k= 30, final_k=20, batch_size=5, batch_overlap=0)


    reranked_df = reranker.rerank_all_users(checkpoint_path=file_name)
    reranked_df.to_csv(file_name, index=False)
    # recs_df.to_csv(f"{method}_recommendations_LLM_Heap.output_sample", index=False)

    print("Saved reranked recommendations")