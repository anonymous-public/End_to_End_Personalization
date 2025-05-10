import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union


class RecEvaluator:
    def __init__(
            self,
            test_df: pd.DataFrame,
            recommendations_df: pd.DataFrame,
            user_id_col: str = "user_id",
            item_id_col: str = "movie_id",
            rating_col: str = "rating",
            rank_col: str = "recommendation_rank",
            module_source_col: Optional[str] = None,
            relevance_threshold: float = 3.0
    ):
        """
        Initialize the RecEvaluator with test and recommendation data.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame containing actual user-item interactions and their ratings.
        recommendations_df : pd.DataFrame
            DataFrame containing recommended items for each user with a ranking.
            May optionally contain multiple recommender modules identified by 'module_source_col'.
        user_id_col : str, optional
            Name of the user ID column.
        item_id_col : str, optional
            Name of the item ID column.
        rating_col : str, optional
            Name of the rating column in test_df.
        rank_col : str, optional
            Name of the ranking column in recommendations_df.
        module_source_col : str, optional
            Column that indicates the "module" or "algorithm" that produced the recommendation.
            If provided, separate evaluations will be computed by module.
        relevance_threshold : float, optional
            Threshold to consider an item as relevant.
        """
        self.test_df = test_df
        self.recommendations_df = recommendations_df
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col
        self.rank_col = rank_col
        self.module_source_col = module_source_col
        self.relevance_threshold = relevance_threshold
        self.item_popularity = None
        self.total_interactions = None

        self._prepare_data()

    def _prepare_data(self):
        """Prepare internal data structures for faster computation."""
        # 1. Filter test data for relevant items (ratings >= threshold).
        self.test_relevant = self.test_df[self.test_df[self.rating_col] >= self.relevance_threshold]

        # 2. For each user, get the set of relevant items.
        self.user_relevant_items = (
            self.test_relevant.groupby(self.user_id_col)[self.item_id_col]
            .apply(set)
            .to_dict()
        )

        # 3. For each user, get the full set of test items (useful if you want coverage-based metrics).
        self.user_all_items = (
            self.test_df.groupby(self.user_id_col)[self.item_id_col]
            .apply(set)
            .to_dict()
        )

        # 4. Build a dict of user -> list of (item, rank, [optional module]) from the recommendations.
        if self.module_source_col and self.module_source_col in self.recommendations_df.columns:
            # We have multiple modules. Group by both user and module.
            self.user_recommendations = {}
            for (user, module), group_df in (
                self.recommendations_df
                .sort_values([self.user_id_col, self.module_source_col, self.rank_col])
                .groupby([self.user_id_col, self.module_source_col])
            ):
                self.user_recommendations.setdefault(user, {})[module] = list(zip(
                    group_df[self.item_id_col],
                    group_df[self.rank_col]
                ))
        else:
            # Single module scenario: group only by user.
            self.user_recommendations = (
                self.recommendations_df
                .sort_values([self.user_id_col, self.rank_col])
                .groupby(self.user_id_col)[[self.item_id_col, self.rank_col]]
                .apply(lambda x: list(zip(x[self.item_id_col], x[self.rank_col])))
                .to_dict()
            )

        # 5. Find users in both test and recommendations.
        self.users_in_test = set(self.user_all_items.keys())

        if isinstance(self.user_recommendations, dict):
            # Two possible structures:
            # (A) user_recommendations[user] -> list[(item, rank)] if no module_source
            # (B) user_recommendations[user][module] -> list[(item, rank)] if module_source_col is present

            # Identify all users in rec
            if self.module_source_col and isinstance(next(iter(self.user_recommendations.values()), {}), dict):
                # We have multi-module data
                users_in_rec = set(self.user_recommendations.keys())
                # Intersection
                self.evaluation_users = list(self.users_in_test.intersection(users_in_rec))
            else:
                # Single module scenario
                users_in_rec = set(self.user_recommendations.keys())
                self.evaluation_users = list(self.users_in_test.intersection(users_in_rec))
        else:
            # Should never happen in normal usage, but just in case
            self.evaluation_users = []

        # Compute item popularity (frequency) from test_df
        item_counts = self.test_df[self.item_id_col].value_counts()
        self.item_popularity = item_counts.to_dict()
        self.total_interactions = item_counts.sum()

    # -------------------------------------------------------------------------
    # Basic retrieval metrics
    # -------------------------------------------------------------------------
    def precision_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute Precision@K across all users (optionally for a given module)."""
        precisions = []

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            if not recommended_items:
                continue

            rel_items = self.user_relevant_items.get(user, set())
            hit_count = len(set(recommended_items) & rel_items)
            precisions.append(hit_count / len(recommended_items))

        return np.mean(precisions) if precisions else 0.0

    def recall_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute Recall@K across all users (optionally for a given module)."""
        recalls = []

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            rel_items = self.user_relevant_items.get(user, set())
            if not rel_items:
                # no relevant items for this user in test -> skip
                continue

            hit_count = len(set(recommended_items) & rel_items)
            recalls.append(hit_count / len(rel_items))

        return np.mean(recalls) if recalls else 0.0

    def f1_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute F1@K across all users (the harmonic mean of precision@K and recall@K)."""
        precision = self.precision_at_k(k, module=module)
        recall = self.recall_at_k(k, module=module)
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _dcg_at_k(self, recommended_items: List[int], relevant_items: set, k: int) -> float:
        """Compute Discounted Cumulative Gain at K."""
        dcg = 0.0
        for idx, item in enumerate(recommended_items[:k], start=1):
            rel = 1.0 if item in relevant_items else 0.0
            dcg += rel / np.log2(idx + 1)
        return dcg

    def ndcg_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute NDCG@K across all users (optionally for a given module)."""
        ndcgs = []

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            if not recommended_items:
                continue

            rel_items = self.user_relevant_items.get(user, set())

            dcg = self._dcg_at_k(recommended_items, rel_items, k)

            # Ideal scenario: put all relevant items in the first positions
            ideal_list = list(rel_items)
            ideal_list = ideal_list[:min(len(ideal_list), k)]
            idcg = self._dcg_at_k(ideal_list, rel_items, k)

            if idcg > 0:
                ndcgs.append(dcg / idcg)

        return np.mean(ndcgs) if ndcgs else 0.0

    def mrr_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute MRR@K (Mean Reciprocal Rank) across all users."""
        mrrs = []

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            rel_items = self.user_relevant_items.get(user, set())

            # Find the first relevant item in the ranked list
            rr = 0.0
            for idx, item in enumerate(recommended_items, start=1):
                if item in rel_items:
                    rr = 1.0 / idx
                    break
            if rr > 0:
                mrrs.append(rr)

        return np.mean(mrrs) if mrrs else 0.0

    def average_precision_at_k(self, k: int, module: Optional[str] = None) -> float:
        """Compute MAP@K (Mean Average Precision) across all users."""
        ap_values = []

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            rel_items = self.user_relevant_items.get(user, set())
            if not rel_items:
                # no relevant items for this user
                continue

            hits = 0
            precision_sum = 0.0
            for idx, item in enumerate(recommended_items, start=1):
                if item in rel_items:
                    hits += 1
                    precision_sum += hits / idx
            if hits > 0:
                ap_values.append(precision_sum / len(rel_items))

        return np.mean(ap_values) if ap_values else 0.0

    # -------------------------------------------------------------------------
    # Coverage metrics
    # -------------------------------------------------------------------------
    def user_coverage_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Fraction of users who receive at least one relevant recommendation among
        users who actually have relevant items.
        """
        covered_users = 0
        total_users_with_relevant = 0

        for user in self.evaluation_users:
            rel_items = self.user_relevant_items.get(user, set())
            if not rel_items:
                continue  # skip users with no relevant items at all
            total_users_with_relevant += 1

            recommended_items = self._get_top_k_items(user, k, module)
            # if there's at least one relevant item in recommended
            if len(set(recommended_items) & rel_items) > 0:
                covered_users += 1

        return covered_users / total_users_with_relevant if total_users_with_relevant > 0 else 0.0

    def item_coverage_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Fraction of distinct items recommended (at rank <= k) out of all possible items
        in the test set.
        """
        recommended_set = set()
        all_items_in_test = set(self.test_df[self.item_id_col].unique())

        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            recommended_set.update(recommended_items)

        return len(recommended_set) / len(all_items_in_test) if len(all_items_in_test) > 0 else 0.0

    def average_popularity_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Compute the average popularity of the top-k recommended items.
        A lower average popularity indicates higher novelty.
        Popularity is derived from the frequency of item appearances in test_df.
        """
        if not hasattr(self, "item_popularity"):
            raise ValueError(
                "Item popularity was not computed. Make sure you have updated _prepare_data()."
            )

        user_popularities = []
        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            if not recommended_items:
                continue

            # Average popularity of recommended items
            pop_sum = 0
            for item in recommended_items:
                pop_sum += self.item_popularity.get(item, 0)

            avg_pop = pop_sum / len(recommended_items)
            user_popularities.append(avg_pop)

        return np.mean(user_popularities) if user_popularities else 0.0

    def self_information_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Compute the average self-information (-log2 p(i)) of the top-k recommended items,
        where p(i) = frequency(item i) / total_interactions.
        Higher average self-information indicates more novel (less popular) items.
        """
        if not hasattr(self, "item_popularity") or not hasattr(self, "total_interactions"):
            raise ValueError(
                "Item popularity or total interactions not computed. "
                "Make sure you have updated _prepare_data()."
            )

        user_self_info = []
        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            if not recommended_items:
                continue

            si_sum = 0.0
            for item in recommended_items:
                freq = self.item_popularity.get(item, 0)
                if freq > 0:
                    p = freq / self.total_interactions
                    si_sum += -np.log2(p)
                else:
                    # If item not in test set (freq = 0), skip or treat as very high novelty
                    # For example, you could do:
                    # si_sum += 0  # ignoring it
                    # or set it to some large penalty
                    pass

            avg_si = si_sum / len(recommended_items)
            user_self_info.append(avg_si)

        return np.mean(user_self_info) if user_self_info else 0.0

    def item_fairness_gini_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Compute the Gini index for item recommendation frequencies at rank <= k.
        A lower Gini index indicates more equitable distribution (fairness) across items.
        """
        # 1. Gather the frequency of recommended items across all users
        freq_map = {}
        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            for item in recommended_items:
                freq_map[item] = freq_map.get(item, 0) + 1

        if not freq_map:
            return 0.0  # or np.nan, depending on how you want to handle empty recs

        frequencies = np.array(list(freq_map.values()), dtype=float)
        return self._gini_index(frequencies)

    def item_fairness_entropy_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Compute the entropy (base 2) of the item recommendation frequency distribution.
        A higher entropy indicates a more uniform distribution (fairness) across items.
        """
        freq_map = {}
        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            for item in recommended_items:
                freq_map[item] = freq_map.get(item, 0) + 1

        if not freq_map:
            return 0.0  # or np.nan

        frequencies = np.array(list(freq_map.values()), dtype=float)
        total = frequencies.sum()
        p = frequencies / total

        # Entropy in base 2: - sum(p_i * log2(p_i))
        entropy = -np.sum(p * np.log2(p))
        return entropy

    def user_fairness_variance_at_k(self, k: int, module: Optional[str] = None) -> float:
        """
        Compute the variance in the number of relevant recommendations per user.
        A lower variance indicates a more equitable (fair) distribution of relevance across users.
        """
        relevant_counts = []
        for user in self.evaluation_users:
            recommended_items = self._get_top_k_items(user, k, module)
            rel_items = self.user_relevant_items.get(user, set())
            # Count how many recommended items are relevant
            hit_count = len(set(recommended_items) & rel_items)
            relevant_counts.append(hit_count)

        if not relevant_counts:
            return 0.0

        return float(np.var(relevant_counts, ddof=1))  # sample variance

    # ------------------------------------------------------------------------------
    # 3) Add private helper methods for Gini index and (optionally) for Entropy,
    #    though entropy is simple enough to inline. For Gini:
    # ------------------------------------------------------------------------------
    def _gini_index(self, values: np.ndarray) -> float:
        """
        Compute the Gini index for a list/array of values.
        Formula reference (one of several):
          Gini = sum_i( sum_j( |x_i - x_j| ) ) / (2 * n * sum_i(x_i))
        """
        if len(values) == 0:
            return 0.0

        sorted_vals = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(sorted_vals)
        # This uses a known simplified expression for Gini with sorted data:
        gini = (n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n
        return gini

    # -------------------------------------------------------------------------
    # Public entry points for batch evaluations
    # -------------------------------------------------------------------------
    def evaluate_all_metrics(
            self,
            k_values: List[int],
            module: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate all relevant metrics at given k values and return a DataFrame.
        If 'module' is None and multiple modules exist, it will combine them
        (all recommendations aggregated). Otherwise, set 'module' to a valid module name.

        Parameters
        ----------
        k_values : list of int
            Values of K to evaluate metrics at.
        module : str, optional
            If multiple modules are present, you can evaluate just one module's performance.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: K, Precision, Recall, F1, NDCG, MRR, MAP, UserCoverage, ItemCoverage
        """
        results = []
        for k in k_values:
            precision = self.precision_at_k(k, module)
            recall = self.recall_at_k(k, module)
            f1 = self.f1_at_k(k, module)
            ndcg = self.ndcg_at_k(k, module)
            mrr = self.mrr_at_k(k, module)
            ap = self.average_precision_at_k(k, module)
            user_cov = self.user_coverage_at_k(k, module)
            item_cov = self.item_coverage_at_k(k, module)
            novelty_avg_pop = self.average_popularity_at_k(k, module)
            novelty_si = self.self_information_at_k(k, module)
            fairness_gini = self.item_fairness_gini_at_k(k, module)
            user_var = self.user_fairness_variance_at_k(k, module)

            results.append({
                "K": k,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": ap,
                "ItemCoverage": item_cov,
                "Novelty": novelty_avg_pop,
                "ItemFairness": fairness_gini,
            })

        df_results = pd.DataFrame(results)
        return df_results

    def evaluate_all_metrics_for_all_modules(self, k_values: List[int]) -> pd.DataFrame:
        """
        If there are multiple modules in recommendations, this method evaluates
        each module separately (plus an 'ALL' aggregated) over the specified K values.

        Parameters
        ----------
        k_values : list of int
            Values of K to evaluate metrics at.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: [Module, K, Precision, Recall, F1, NDCG, MRR, MAP, UserCoverage, ItemCoverage]
        """
        # If there's no module_source_col, just run evaluate_all_metrics once
        if not self.module_source_col or self.module_source_col not in self.recommendations_df.columns:
            single_result = self.evaluate_all_metrics(k_values, module=None)
            single_result.insert(0, "Module", "ALL")
            return single_result

        # Identify all unique modules
        all_modules = self.recommendations_df[self.module_source_col].unique()

        # Evaluate each module separately
        all_results = []
        for mod in all_modules:
            mod_df = self.evaluate_all_metrics(k_values, module=mod)
            mod_df.insert(0, "Module", mod)
            all_results.append(mod_df)

        # Optionally, evaluate an "ALL" aggregator:
        # i.e., pretend all recommended items are from the same single system
        # This can be done if it makes sense to aggregate them.
        all_results_df = self.evaluate_all_metrics(k_values, module=None)
        all_results_df.insert(0, "Module", "ALL")
        all_results.append(all_results_df)

        # Combine
        final = pd.concat(all_results, ignore_index=True)
        return final

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def plot_metrics(
            self,
            results_df: pd.DataFrame,
            metrics: Optional[List[str]] = None,
            title: Optional[str] = None,
            save_path: Optional[str] = None
    ):
        """
        Plot selected metrics over K from a given results DataFrame.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame containing at least ['K'] and the metrics to plot.
        metrics : list of str, optional
            Metrics to plot. Default: ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP"]
        title : str, optional
            Plot title.
        save_path : str, optional
            If provided, save the plot to this path.
        """
        if metrics is None:
            metrics = ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP", "ItemCoverage", "Novelty", "ItemFairness"]

        # Basic validation
        for metric in metrics:
            if metric not in results_df.columns:
                raise ValueError(f"{metric} not found in the results_df columns.")

        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(results_df["K"], results_df[metric], marker='o', label=metric)

        plt.xlabel("K")
        plt.ylabel("Metric Value")
        plt.title(title if title else "Recommendation Metrics Over K")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_metrics_by_module(
            self,
            results_df: pd.DataFrame,
            metrics: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Plot selected metrics over K for each module separately (faceted by module).

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame containing columns ["Module", "K"] and the metrics to plot.
        metrics : list of str, optional
            Metrics to plot. Default: ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP"]
        save_path : str, optional
            If provided, save the resulting figure to this path.
        """
        if self.module_source_col not in results_df.columns and "Module" not in results_df.columns:
            raise ValueError("No 'Module' column found in the results. Cannot plot by module.")

        if metrics is None:
            metrics = ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP"]

        unique_modules = results_df["Module"].unique()
        n_mods = len(unique_modules)
        fig, axes = plt.subplots(n_mods, 1, figsize=(8, 4 * n_mods), sharex=True)

        if n_mods == 1:
            axes = [axes]  # make it iterable

        for ax, mod in zip(axes, unique_modules):
            sub_df = results_df[results_df["Module"] == mod]
            for metric in metrics:
                if metric not in sub_df.columns:
                    continue
                ax.plot(sub_df["K"], sub_df[metric], marker='o', label=metric)
            ax.set_title(f"Module: {mod}")
            ax.set_xlabel("K")
            ax.set_ylabel("Metric Value")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # -------------------------------------------------------------------------
    # Saving results
    # -------------------------------------------------------------------------
    def save_results_to_csv(self, results_df: pd.DataFrame, csv_path: str):
        """
        Save the evaluation results to a CSV file.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with the evaluation metrics.
        csv_path : str
            Path to the output CSV file.
        """
        results_df.to_csv(csv_path, index=False)

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------
    def _get_top_k_items(self, user: Union[int, str], k: int, module: Optional[str]) -> List[int]:
        # Get all items that appear in the test set for this user
        user_test_items = self.user_all_items.get(user, set())

        # Multi-module scenario
        if (
                self.module_source_col
                and user in self.user_recommendations
                and isinstance(self.user_recommendations[user], dict)
        ):
            if module is not None and module in self.user_recommendations[user]:
                # Only keep recommended items that are in the user's test set and within rank <= k
                return [
                    item
                    for item, rank in self.user_recommendations[user][module]
                    if rank <= k and item in user_test_items
                ]
            else:
                # Combine all modules, filter by rank <= k, and keep only items that appear in user's test set
                combined_items = []
                for mod, items_ranks in self.user_recommendations[user].items():
                    combined_items.extend(
                        [
                            (item, rank)
                            for item, rank in items_ranks
                            if rank <= k and item in user_test_items
                        ]
                    )
                # Re-sort by rank and pick top-k
                combined_items.sort(key=lambda x: x[1])
                return [x[0] for x in combined_items[:k]]

        # Single-module scenario
        if user not in self.user_recommendations:
            return []
        return [
            item
            for item, rank in self.user_recommendations[user]
            if rank <= k and item in user_test_items
        ]
