## Two-layer Hybrid Recommender System in retail

### Contents

**Stack:**

- 1-st layer: NLP, Implicit, ItemItemRecommender, ALS, sklearn, pandas, numpy, matplotlib
- 2-nd layer: CatBoost, LightGBM


**Data:** from [Retail X5 Hero Competition](https://retailhero.ai/c/recommender_system/overview)


**Task:** 
Create two-layer hybrid recommender system for retail data and evaluate it by custom **Precision@k**.


**Steps**:

1. Prepare data:  prefiltering
2. Matching model (initialize MainRecommender 1-st layer model as baseline)
3. Evaluate Top@k Recall
4. Ranking model (choose 2-nd layer model)
5. Feature engineering for ranking


### User guide

Please, open [train.ipynb](https://github.com/hildar/RecSys-Retail/blob/main/train.ipynb) Jupiter notebook file and explore how to create *Recommender system* step-by-step.

Project has next few steps:

#### 1. Prepare data

First is looking at datasets and prefiltering data

![data image](img/data.png)


#### 2. Matching model

Learn first-layer model as baseline. In `MainRecommender` class we have two base models from implicit lib - `ItemItemRecommender` and `AlternatingLeastSquares`:

```
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
```

`ALS` used to find similar users, items and als recommendations. `ItemItemRecommender` used to find own item recommendations among user's purchases.

#### 3. Evaluate Top@k Recall

For first-layer model we have taken Recall metric because it is show the proportion of correct answers from real purchases. With this approach we going to significantly cut dataset size for second-layer model.

Here we are evaluating different types of recommendations:

<img src="img/types_recs.png" alt="drawing" width="600"/>

And are selecting optimal value of Recall:

![recall](img/recall.png)



