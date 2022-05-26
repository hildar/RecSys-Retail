"""
Baseline for RecSys

"""
import pandas as pd
import numpy as np


# For sparse matrix
from scipy.sparse import csr_matrix


# Matrix factorization
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # for own recommend
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """
    Baseline
    Get base recomendations from ALS or k-nearest algorithms
    Input
    -----
    user_item_matrix: pd.DataFrame
        Matrix user-item conversations
    """

    def __init__(self, data: pd.DataFrame, weighting: bool = True):

        # Top popular purchases of each user
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Top popular purchases of all dataset
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # Get sparse matrix
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        
        # Get user-item ids map dictionary
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        # Weighting matrix for better result
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        # Learn two type of models
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data: pd.DataFrame):
        """Prepare sparse user-item matrix"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',  # variable
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # necessary matrix type for implicit 

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Prepare ids dictionaries"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Learn model that get item recommendations among user's purchases"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Learn ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Update dicts if new user/item has added"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Find new similar item to item_id"""

        # Get two recs and choose only second because first is self item_id
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Extend recs with top popular if num of items less than N"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Get top-N recommendations from standart algorithms of implicit lib"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Number of recommendations != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):       
        """Get top-N ALS recommendations"""

        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Get top-N own item recommendations among user's purchases"""

        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user_id, N=5):
        """Get top-N items similar to user's top popular self purchases"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Number of recommendations != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=5):
        """Get top-N items among similar users purchases"""

        res = []

        # Find top-N similar users (get it with 10 reserve users)
        similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N + 11)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # delete self user's id from query

        # Get one own recommendation for each similar user
        for _user_id in similar_users:
            _rec = self.get_own_recommendations(_user_id, N=1)
            # add only unique item
            if _rec not in res:
                res.extend(_rec)

        # Cut if redundand
        res = res[:N]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Number of recommendations != {}'.format(N)
        return res
