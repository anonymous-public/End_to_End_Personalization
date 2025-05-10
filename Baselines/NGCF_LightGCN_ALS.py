import time
import pandas as pd
import tensorflow as tf
from libreco.algorithms import ALS, NGCF, LightGCN
from libreco.data import DatasetPure, split_by_ratio


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


def get_model_recommendations_df(model, model_name, trainset_df, n_rec=50, buffer_factor=2):
    """
    For each user, call recommend_user(...) to get top-n_rec items.
    Build and return a DataFrame with columns:
        user_id | movie_id | recommendation_rank | module_source
    """
    # Get all unique users
    all_users = trainset_df["user"].unique().tolist()

    # Dictionary to store seen items for each user
    user_seen_items = trainset_df.groupby("user")["item"].apply(set).to_dict()

    # Generate more recommendations upfront (buffer_factor * n_rec)
    rec_dict = model.recommend_user(user=all_users, n_rec=buffer_factor * n_rec)

    df_rows = []
    for user_id, items in rec_dict.items():
        # Get the set of items the user has already seen
        seen_items = user_seen_items.get(user_id, set())

        # Filter out seen items from the recommendations
        filtered_items = [item_id for item_id in items if item_id not in seen_items]

        # Ensure we have exactly n_rec recommendations
        filtered_items = filtered_items[:n_rec]

        # Add the filtered recommendations to the DataFrame rows
        for rank, item_id in enumerate(filtered_items, start=1):
            df_rows.append((user_id, item_id, rank, model_name))

    # Create the DataFrame
    recommendations_df = pd.DataFrame(
        df_rows,
        columns=["user_id", "movie_id", "recommendation_rank", "module_source"]
    )

    return recommendations_df


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

    # ------------------------------------------------
    # ------------------------------------------------
    # 1) Rename columns
    trainset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label"
        },
        inplace=True
    )

    testset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label"
        },
        inplace=True
    )

    # 2) Re-order columns if needed (e.g., drop the timestamp or put it last)
    trainset_df = trainset_df[["user", "item", "label", "timestamp"]]
    testset_df = testset_df[["user", "item", "label", "timestamp"]]

    # Suppose after renaming columns to 'user' and 'item'
    train_users = set(trainset_df["user"].unique())
    train_items = set(trainset_df["item"].unique())

    # Filter test so it only has user/item that appear in train
    testset_df = testset_df[
        testset_df["user"].isin(train_users) & testset_df["item"].isin(train_items)
        ]

    # ------------------------------------------------
    # ------------------------------------------------

    # Build train and eval sets
    train_data, eval_data = split_by_ratio(trainset_df, test_size=0.1)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    print(data_info)

    data_info_ngcf = data_info
    data_info_lightgcn = data_info
    data_info_als = data_info

    # =========================== load model ==============================

    all_train_users = trainset_df["user"].unique()

    metrics = [
        "loss",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "map",
        "ndcg",
    ]

    # ------------------------------------------------------------
    # NGCF
    # ------------------------------------------------------------
    reset_state("NGCF")
    ngcf = NGCF(
        task="ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=64,
        n_epochs=100,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        node_dropout=0.1,
        message_dropout=0.1,
        hidden_units=(64, 64, 64),
        device="cuda",
    )
    ngcf.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    # print("NGCF prediction (user=1, item=2333): ", ngcf.predict(user=1, item=2333))
    # print("NGCF recommendation (user=1): ", ngcf.recommend_user(user=1, n_rec=7))

    # save data_info, specify model save folder
    data_info_ngcf.save(path=f"{data}/", model_name="ngcf")
    ngcf.save(path=f"{data}/", model_name="ngcf", manual=True, inference_only=True)


    # 2) NGCF recommendations
    ngcf_recs_df = get_model_recommendations_df(
        model=ngcf,
        model_name="NGCF",
        trainset_df=trainset_df,
        n_rec=50  # or whatever number of recommendations you prefer
    )

    # 5) Convert columns to the desired data types
    ngcf_recommendations_df = ngcf_recs_df.astype({
        'user_id': 'int',
        'movie_id': 'int',
        'recommendation_rank': 'int',
        'module_source': 'str'
    })

    ngcf_recommendations_df.to_csv(f"ngcf_result_{data}.csv")
    print("****************** NGCF competeted ******************")

    # ------------------------------------------------------------
    # LightGCN
    # ------------------------------------------------------------
    reset_state("LightGCN")
    lightgcn = LightGCN(
        task="ranking",
        data_info=data_info,
        loss_type="bpr",
        embed_size=64,
        n_epochs=100,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.1,
        n_layers=3,
        device="cuda",
    )
    lightgcn.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )

    # save data_info, specify model save folder
    data_info_lightgcn.save(path=f"{data}/", model_name="lightgcn")
    lightgcn.save(path=f"{data}/", model_name="lightgcn", manual=True, inference_only=True)

    # 3) LightGCN recommendations
    lightgcn_recs_df = get_model_recommendations_df(
        model=lightgcn,
        model_name="LightGCN",
        trainset_df=trainset_df,
        n_rec=50
    )

    lightgcn_recommendations_df = lightgcn_recs_df.astype({
        'user_id': 'int',
        'movie_id': 'int',
        'recommendation_rank': 'int',
        'module_source': 'str'
    })

    lightgcn_recommendations_df.to_csv(f"lightgcn_result_{data}.csv")
    print("****************** LightGCN competeted ******************")

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    reset_state("ALS")
    als = ALS(
        "ranking",
        data_info,
        embed_size=64,
        n_epochs=100,
        reg=5.0,
        alpha=10,
        use_cg=False,
        n_threads=1,
        seed=42,
    )
    als.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics,
    )

    # save data_info, specify model save folder
    data_info_als.save(path=f"{data}/", model_name="als")
    als.save(path=f"{data}/", model_name="als", manual=True, inference_only=True)

    # 3) LightGCN recommendations
    als_recs_df = get_model_recommendations_df(
        model=als,
        model_name="ALS",
        trainset_df=trainset_df,
        n_rec=50
    )

    als_recommendations_df = als_recs_df.astype({
        'user_id': 'int',
        'movie_id': 'int',
        'recommendation_rank': 'int',
        'module_source': 'str'
    })

    als_recommendations_df.to_csv(f"als_result_{data}.csv")
    print("****************** ALS competeted ******************")

    # ------------------------------------------------------------
    # ------------------------------------------------------------





    # 4) Concatenate both
    # recommendations_df = pd.concat([ngcf_recs_df, lightgcn_recs_df], ignore_index=True)
    #print("\nFinal Recommendations DataFrame:\n", recommendations_df.head(20))
