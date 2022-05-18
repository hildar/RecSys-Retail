"""
Filters for RecSys

"""

import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    """Prefilter items and take top popular"""

    # Delete rare categories (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Delete cheap items (non profit). Price one purchase from mailing is 1 dollar
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Delete expensive items
    data = data[data['price'] < 50]

    # Get top popular items
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Set a fake id for non popular items
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(user_id, recommednations):
    """Postfilter items after fit models"""

    # What time to show?
    # How often to show?
    pass