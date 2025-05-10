import pandas as pd

from GCF import GCF
from VGCF import VGCF
# from VGCF_new_loss import VGCF
from Vectorized_Clustered_KNN import ClusteredKNN
from UserBasedRecommender import UserBasedRecommender
# from VNMF import VNMF
from Embedding.embedding_layer import EmbeddingLayer
from Evaluation.RecEvaluator import RecEvaluator

# from baselines.NGCF import NGCF

if __name__ == "__main__":
    data = "1m"
    Path = f"../Dataset/{data}/train_test_sets"

    # Load the ratings data
    trainset_df = pd.read_csv(f'{Path}/u1.base',
                              sep='\t',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              encoding='latin-1')

    testset_df = pd.read_csv(f'{Path}/u1.test',
                             sep='\t',
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             encoding='latin-1')

    users_df = pd.read_csv(f"../Dataset/preprocessed/users_structured_{data}.csv")
    content_df = pd.read_csv(f"../Dataset/preprocessed/movies_structured_{data}.csv")

    # Filter rows where 'summary' and 'movie_info' are not null
    users_df = users_df.dropna(subset=["user_profile"])
    content_df = content_df.dropna(subset=["movie_info"])

    embedd = EmbeddingLayer()

    users_df = embedd.convert_to_embeddings(df=users_df, col="user_profile")
    content_df = embedd.convert_to_embeddings(df=content_df, col="movie_info")

    # ------------------------------------------------------------
    # clustered_KNN

    # recommender = ClusteredKNN(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                              content_id_column="movie_id",
    #                              user_attribute_column="summary_vector",
    #                              content_attribute_column="movie_info_vector")
    #
    # knn_recommendations = recommender.get_recommendations()
    # knn_recommendations.to_csv(f"CVknn_recommendations_{data}.csv")
    # print("done")

    # ------------------------------------------------------------
    # user_based profile

    # recommender = UserBasedRecommender(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                              content_id_column="movie_id",
    #                              user_attribute_column="summary_vector",
    #                              content_attribute_column="movie_info_vector")
    #
    # user_based_recommendations = recommender.get_recommendations()
    # user_based_recommendations.to_csv(f"UB_recommendations_{data}.csv")
    # print("done")

    # ------------------------------------------------------------
    # vectorized VGCF

    recommender = VGCF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    recommender.set_column_names(user_id_column="user_id",
                                content_id_column="movie_id",
                                user_attribute_column="user_profile_vector",
                                content_attribute_column="movie_info_vector")

    vgcf_recommendations = recommender.get_recommendations()
    print("done")
    print(vgcf_recommendations.head())
    vgcf_recommendations.to_csv(f"vgcf_new_recommendations_{data}.csv")

    # ------------------------------------------------------------
    # vectorized VGCF

    # recommender = GCF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                             content_id_column="movie_id",
    #                             user_attribute_column="summary_vector",
    #                             content_attribute_column="movie_info_vector")
    #
    # gat_recommendations = recommender.get_recommendations()
    # print("done")
    # print(gat_recommendations.head())
    # gat_recommendations.to_csv(f"gat_recommendations_{data}.csv")

    # ------------------------------------------------------------

    # VNMF profile
    #
    # recommender = VNMF(users_df=users_df, content_df=content_df, interactions_df=trainset_df)
    # recommender.set_column_names(user_id_column="user_id",
    #                             content_id_column="movie_id",
    #                             user_attribute_column="summary_vector",
    #                             content_attribute_column="movie_info_vector")
    #
    # recommender.train_model()
    #
    # vnmf_recommendations = recommender.get_recommendations()
    # print("done")
    # print(vnmf_recommendations.head())

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Instantiate the evaluator
    rec_evaluator = RecEvaluator(
        test_df=testset_df,
        recommendations_df=vgcf_recommendations,
        user_id_col="user_id",
        item_id_col="movie_id",
        rating_col="rating",  # Column in 'testset_df' for ratings
        rank_col="recommendation_rank",  # Column in 'recommendations' for ranking
        relevance_threshold=4.0  # Typically 4.0+ means "relevant" in MovieLens
    )

    # 4. Evaluate at several K values (commonly used in academic papers)
    k_values = [1, 5, 10, 20, 30, 40, 50]

    # Compute all metrics
    results_df = rec_evaluator.evaluate_all_metrics(k_values)

    # 5. Print or inspect the results
    print("Evaluation Results:\n", results_df)

    # 6. Plot the metrics and store the plot
    #    By default, it plots Precision, Recall, NDCG, MRR, MAP
    #    but we can also specify exactly which to plot:
    metrics_to_plot = ["Precision", "Recall", "NDCG", "MRR", "MAP"]
    rec_evaluator.plot_metrics(results_df=results_df, metrics=metrics_to_plot, save_path="evaluation_plot.png")

    # 7. Save the numeric results to a CSV file
    rec_evaluator.save_results_to_csv(results_df=results_df, csv_path="evaluation_results.csv")

    # Now you have:
    # - A CSV file with the numeric results
    # - A plot image (e.g., "evaluation_plot.png") showing metric curves vs. K
