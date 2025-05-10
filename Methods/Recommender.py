"""
Explanation
"""
import gc

# ----------------------------------------------------------------------------------------------------
# Importing necessary libraries

from abc import ABC, abstractmethod

# ----------------------------------------------------------------------------------------------------
# Define the DataLoader class
class Recommender(ABC):
    """
    This is an abstract class different recommendation methods
    """

    def __init__(self, content_df, interactions_df, users_df):
        # dataframes
        self.content_df = content_df
        self.interactions_df = interactions_df
        self.users_df = users_df

        # dataframe containing recommendations
        self.recommendations_df = None

        self.content_dict = None  # content_id and feature vector dictionary
        self.all_reduced_vectors = None  # for visualization purposes

        # necessary information which we calculate in the class
        self.user_profiles = {}
        self.cosine_scores = {}
        self.all_vectors = None

        self.total_recommendations = 50

        # column names: will be set by user
        self.user_id_column = None
        self.content_id_column = None
        self.user_attribute_column = None
        self.content_attribute_column = None

    def set_column_names(self, user_id_column, content_id_column, user_attribute_column, content_attribute_column):
        self.user_id_column = user_id_column
        self.content_id_column = content_id_column
        self.user_attribute_column = user_attribute_column
        self.content_attribute_column = content_attribute_column

    @abstractmethod
    def get_recommendations(self):
        pass





