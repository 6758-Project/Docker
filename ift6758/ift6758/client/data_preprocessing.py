import pandas as pd


TRAIN_COLS_PART_4 = [
    "game_sec",
    "period",
    "secondary_type",
    "coordinate_x",
    "coordinate_y",
    "distance_from_net",
    "angle",
    "prev_event_type",
    "angle_between_prev_event",
    "distance_from_prev_event",
    "prev_event_time_diff",
    "speed",
    "is_rebound",
    "rebound_angle",
    "is_empty_net",
]

INFREQUENT_STOPPAGE_EVENTS = [
    "PERIOD_START",
    "PERIOD_READY",
    "PERIOD_END",
    "SHOOTOUT_COMPLETE",
    "PERIOD_OFFICIAL",
    "GAME_OFFICIAL",
    "PENALTY",
    "GOAL",
    "CHALLENGE",
]

SHAP_COLS = [
    "distance_from_net",
    "is_rebound",
    "prev_event_SHOT",
    "prev_event_time_diff",
    "angle",
    "is_empty_net",
    "shot_Snap Shot",
    "shot_Slap Shot",
    "distance_from_prev_event",
    "coordinate_y",
    "prev_event_HIT",
]

LASSO_COLS = [
    "coordinate_x",
    "coordinate_y",
    "distance_from_net",
    "angle",
    "angle_between_prev_event",
    "distance_from_prev_event",
    "prev_event_time_diff",
    "speed",
    "is_rebound",
    "rebound_angle",
    "is_empty_net",
    "shot_Backhand",
    "shot_Tip-In",
    "shot_Wrist Shot",
    "prev_event_FACEOFF",
    "prev_event_GIVEAWAY",
    "prev_event_HIT",
    "prev_event_SHOT",
]


def preprocess_xgboost_shap(x_data):

    x_data["game_id"] = x_data["game_id"].astype(int)
    x_data = x_data[TRAIN_COLS_PART_4]
    x_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    x_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    x_data = pd.get_dummies(x_data, ["shot", "prev_event"])

    x_data = x_data[SHAP_COLS]

    return x_data


def preprocess_xgboost_lasso(x_data):

    x_data["game_id"] = x_data["game_id"].astype(int)
    x_data = x_data[TRAIN_COLS_PART_4]
    x_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    x_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    x_data = pd.get_dummies(x_data, ["shot", "prev_event"])

    x_data = x_data[LASSO_COLS]

    return x_data


def preprocess_xgboost_non_corr_feats(x_data):

    x_data["game_id"] = x_data["game_id"].astype(int)
    x_data = x_data[TRAIN_COLS_PART_4]
    x_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    x_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    x_data = pd.get_dummies(x_data, ["shot", "prev_event"])

    # the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = x_data.columns.difference(redundant_feats)
    X_data = x_data[selected_feats]

    return X_data
