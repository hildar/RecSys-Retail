## Two-layer Hybrid Recommender System in retail

### Contents

**Stack:**

- 1-st layer: NLP, Implicit, ItemItemRecommender, ALS, sklearn, pandas, numpy, matplotlib
- 2-nd layer: CatBoost, LightGBM


**Data:** from [Retail X5 Hero Competition](https://retailhero.ai/c/recommender_system/overview)


**Task:** 
Create two-layer hybrid recommender system for retail data and evaluate it by custom **Precision@k**.


**Stages**:

1. Prepare data:  prefiltering
2. Matching model (initialize MainRecommender 1-st layer model as baseline)
3. Evaluate Top@k Recall
4. Ranking model (choose 2-nd layer model)
5. Feature engineering for ranking


### User guide

Please, open [train.ipynb](https://github.com/hildar/RecSys-Retail/blob/main/train.ipynb) Jupiter notebook file and explore how to create *Recommender system* step-by-step.
