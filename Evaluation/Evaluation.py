import pandas as pd
import seaborn as sns
from RecEvaluator import RecEvaluator


def load_recommendations(path_dict):
    """
    Load recommendation results from different recommender systems.

    Args:
        path_dict (dict): A dictionary where keys are recommender names and values are file paths.

    Returns:
        dict: A dictionary with recommender names as keys and their corresponding DataFrames as values.
    """
    recommendations = {}
    for name, path in path_dict.items():
        try:
            recommendations[name] = pd.read_csv(path)
            print(f"Loaded recommendations for {name} from {path}.")
        except Exception as e:
            print(f"Error loading recommendations for {name} from {path}: {e}")
    return recommendations


def evaluate_recommenders(test_df, recommendations_dict, k_values, evaluator_params):
    """
    Evaluate multiple recommender systems and collect their performance metrics.

    Args:
        test_df (pd.DataFrame): The test dataset.
        recommendations_dict (dict): Dictionary of recommender names and their recommendation DataFrames.
        k_values (list): List of K values to evaluate.
        evaluator_params (dict): Parameters required to instantiate RecEvaluator.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing evaluation metrics for all recommenders.
    """
    all_results = []

    for name, rec_df in recommendations_dict.items():
        print(f"Evaluating recommender: {name}")
        # Instantiate the evaluator for the current recommender
        rec_evaluator = RecEvaluator(
            test_df=test_df,
            recommendations_df=rec_df,
            user_id_col=evaluator_params['user_id_col'],
            item_id_col=evaluator_params['item_id_col'],
            rating_col=evaluator_params['rating_col'],
            rank_col=evaluator_params['rank_col'],
            relevance_threshold=evaluator_params['relevance_threshold']
        )

        # Compute all metrics for the current recommender
        results_df = rec_evaluator.evaluate_all_metrics(k_values)
        results_df['Recommender'] = name  # Add a column to identify the recommender

        all_results.append(results_df)
        print(f"Completed evaluation for {name}.")

    # Concatenate all results into a single DataFrame
    consolidated_results = pd.concat(all_results, ignore_index=True)
    return consolidated_results


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_all_metrics(consolidated_df,
                              metrics=['Precision', 'Recall', 'NDCG', 'MAP', 'ItemCoverage', 'Novelty'],
                              save_path='evaluation_plots.svg'):
    """
    Plot evaluation metrics (one subplot per metric) for all recommenders in a 2x3 grid.
    Results are saved in SVG format for high-quality vector output.

    Args:
        consolidated_df (pd.DataFrame): DataFrame containing at least the following columns:
                                        ['Recommender', 'K', 'Precision', 'Recall',
                                         'NDCG', 'MAP', 'ItemCoverage', 'Novelty'].
        metrics (list): List of metric names to plot.
        save_path (str): Path to save the SVG plot.

    Example of consolidated_df columns:
        Recommender | K | Precision | Recall | NDCG | MAP | ItemCoverage | Novelty
    """

    # Set a high-quality theme suitable for scientific publications
    sns.set_theme(style="whitegrid", context="talk")

    # Create a figure with 2 rows and 3 columns -> 6 subplots total
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()  # Flatten so we can iterate over them easily

    # Define a color palette (you can change to "Set2", "Paired", etc. as you prefer)
    palette = sns.color_palette("Set2", n_colors=consolidated_df['Recommender'].nunique())

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Plot each metric as a line plot vs. K for each Recommender
        sns.lineplot(
            data=consolidated_df,
            x='K', y=metric,
            hue='Recommender',
            palette=palette,
            marker='o',
            ax=ax
        )

        # Set subplot title and labels
        ax.set_title(metric, fontsize=18, pad=12)
        ax.set_xlabel("K", fontsize=16)
        ax.set_ylabel("Score", fontsize=16)

        # Increase the tick label font size
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Remove the legend from individual subplots
        ax.legend_.remove()

    # Collect handles and labels from the last axes (or any axes) to create a single legend
    handles, labels = ax.get_legend_handles_labels()

    # Create a single legend below all subplots
    fig.legend(
        handles, labels,
        loc='lower center',  # place at the bottom
        ncol=consolidated_df['Recommender'].nunique(),
        frameon=False,
        fontsize=18
    )

    # Adjust the space so the legend fits nicely
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save as SVG to preserve vector quality
    plt.savefig(save_path, format='svg', dpi=300)
    plt.close(fig)
    print(f"Saved evaluation plots to {save_path} (SVG format).")

#
# def plot_all_metrics(consolidated_df,
#                      dataset_name,
#                      metrics=['Precision', 'Recall', 'NDCG', 'MAP'],
#                      save_path='evaluation_plots.svg'):
#     """
#     Plot evaluation metrics (one subplot per metric) for all recommenders in a 1x4 grid.
#     Results are saved in SVG format for high-quality vector output.
#
#     Args:
#         consolidated_df (pd.DataFrame): DataFrame containing at least the following columns:
#                                         ['Recommender', 'K', 'Precision', 'Recall',
#                                          'NDCG', 'MAP', 'ItemCoverage', 'Novelty'].
#         dataset_name (str): Name of the dataset to display as the plot title.
#         metrics (list): List of metric names to plot.
#         save_path (str): Path to save the SVG plot.
#
#     Example of consolidated_df columns:
#         Recommender | K | Precision | Recall | NDCG | MAP | ItemCoverage | Novelty
#     """
#
#     # Set a high-quality theme suitable for scientific publications
#     sns.set_theme(style="whitegrid", context="talk")
#
#     # Create a figure with 1 row and 4 columns -> 4 subplots total
#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))
#     axes = axes.flatten()  # Flatten for easier iteration
#
#     # Define a color palette
#     palette = sns.color_palette("Set2", n_colors=consolidated_df['Recommender'].nunique())
#
#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         # Plot each metric as a line plot vs. K for each Recommender
#         sns.lineplot(
#             data=consolidated_df,
#             x='K', y=metric,
#             hue='Recommender',
#             palette=palette,
#             marker='o',
#             ax=ax
#         )
#
#         # Set subplot title and labels
#         ax.set_title(metric, fontsize=18, pad=12)
#         ax.set_xlabel("K", fontsize=16)
#         ax.set_ylabel("Score", fontsize=16)
#         ax.tick_params(axis='both', which='major', labelsize=16)
#
#         # Remove the legend from individual subplots if it exists
#         if ax.get_legend() is not None:
#             ax.get_legend().remove()
#
#     # Collect handles and labels from the last axes to create a single legend
#     handles, labels = ax.get_legend_handles_labels()
#
#     # Create a single legend below all subplots
#     fig.legend(
#         handles, labels,
#         loc='lower center',
#         ncol=consolidated_df['Recommender'].nunique(),
#         frameon=False,
#         fontsize=18
#     )
#
#     # Add a main title to the figure with the dataset name
#     fig.suptitle(f"Evaluation Metrics on {dataset_name}", fontsize=20)
#
#     # Adjust the space to accommodate the title and the legend without overlapping x-axis labels
#     plt.subplots_adjust(top=0.85, bottom=0.25)
#
#     # Save as SVG to preserve vector quality
#     plt.savefig(save_path, format='svg', dpi=300)
#     plt.close(fig)
#     print(f"Saved evaluation plots to {save_path} (SVG format).")


def main():
    data = "100k"
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

    # testset_df = pd.read_csv(f'../Lusifer/{data}_test_set/enriched_test_set_{data}.csv')

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    #cold start scenario
    #Count the number of interactions per user in the training set
    # user_interaction_counts = trainset_df['user_id'].value_counts()
    #
    # # Identify cold start users: those with fewer than 5 interactions in trainset_df
    # cold_start_users = user_interaction_counts[user_interaction_counts < 20].index.tolist()
    #
    # # Create a new testset_df that only includes rows for these cold start users
    # testset_df = testset_df[testset_df['user_id'].isin(cold_start_users)]

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Load recommendation results for all recommenders
    recommenders_paths = {
        "LLM_rerank_binary": f"../output/{data}/vgcf_new_recommendations_LLM_BST_{data}_k20.csv",
        "Batch_localLLM": f"../output/{data}/vgcf_new_recommendations_localLLM_k20_batch_{data}.csv",
        "rel_score": f"../output/{data}/vgcf_new_recommendations_relevancy_score_k20_{data}.csv",
        "rel_score_4o": f"../output/{data}/vgcf_new_recommendations_relevancy_score_k20_4o_{data}.csv",
        "rel_score_o4": f"../output/{data}/vgcf_new_recommendations_relevancy_score_k20_o4_{data}.csv",
        "Our New Method": f"../output/{data}/vgcf_new_recommendations_{data}.csv",
        # "Our Method": f"../output/{data}/vgcf_recommendations_{data}.csv",
        #"LightGCN": f"../output/{data}/lightgcn_result_{data}.csv",
        "NGCF": f"../output/{data}/ngcf_result_{data}.csv",
        #"ALS": f"../output/{data}/als_result_{data}.csv",
        #"GAT": f"../output/{data}/gat_recommendations_{data}.csv"
    }
    recommendations_dict = load_recommendations(recommenders_paths)
    # ---------------------------------------------------------------------

    # Define evaluator parameters
    evaluator_params = {
        'user_id_col': "user_id",
        'item_id_col': "movie_id",
        'rating_col': "rating",  # Column in 'testset_df' for ratings
        'rank_col': "recommendation_rank",  # Column in 'recommendations' for ranking
        'relevance_threshold': 4.0  # Typically 4.0+ means "relevant" in MovieLens
    }

    # Define K values and metrics to evaluate
    k_values = [1, 5, 10, 15, 20]
    # metrics_to_plot = ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP", "ItemCoverage", "Novelty", "ItemFairness"]
    metrics_to_plot = ["Precision", "Recall", "NDCG", "MAP", "F1", "ItemCoverage"]

    # Evaluate all recommenders
    consolidated_results = evaluate_recommenders(
        test_df=testset_df,
        recommendations_dict=recommendations_dict,
        k_values=k_values,
        evaluator_params=evaluator_params
    )

    # Reorder columns for better readability
    columns_order = ['Recommender', 'K'] + metrics_to_plot
    consolidated_results = consolidated_results[columns_order]

    # # 5. Print or inspect the consolidated results
    # print("Consolidated Evaluation Results:\n", consolidated_results)
    #
    # 6. Plot the metrics for all recommenders and store the plot
    plot_all_metrics(
        consolidated_df=consolidated_results,
        metrics=metrics_to_plot,
        save_path=f"evaluation_plot_comparison_{data}.svg"
    )

    # plotter = Plotter()
    # plotter.plot_metrics(results_df=consolidated_results,
    #                      metrics=metrics_to_plot,
    #                      save_path="evaluation_plot_comparison.png")

    # 7. Save the consolidated numeric results to a CSV file
    consolidated_results.to_csv(f"evaluation_results_comparison_{data}.csv", index=False)
    print(f"Saved consolidated evaluation results to evaluation_results_comparison_{data}.csv")


if __name__ == "__main__":
    main()
