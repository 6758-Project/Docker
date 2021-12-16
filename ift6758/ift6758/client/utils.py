"""
Extracts the events information from the downloaded data with the NHL API.
"""
import numpy as np
import pandas as pd

from datetime import timedelta


## General
PROJECT_NAME = "ift6758-hockey"
WORK_SPACE = "tim-k-lee"

STANDARDIZED_GOAL_COORDINATES = (89, 0)

LABEL_COL = "is_goal"
AVAILABLE_MODELS = {
    "logistic-regression-distance-only": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Distance Only",
        "comet_model_name": "logistic-regression-distance-only",
        "version": "1.0.2",
        "file_name": "LR_distance_only",
    },
    "logistic-regression-angle-only": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Angle Only",
        "comet_model_name": "logistic-regression-angle-only",
        "version": "1.0.3",
        "file_name": "LR_angle_only",
    },
    "logistic-regression-distance-and-angle": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Distance and Angle",
        "comet_model_name": "logistic-regression-distance-and-angle",
        "version": "1.0.2",
        "file_name": "LR_distance_and_angle",
    },
    "xgboost-lasso": {
        "model_type": "xgboost_lasso",
        "model_desc": "XGBoost Model with Lasso",
        "comet_model_name": "xgboost-lasso",
        "version": "1.0.1",
        "file_name": "xgboost_lasso",
    },
    "xgboost-shap": {
        "model_type": "xgboost_SHAP",
        "model_desc": "XGBoost Model with SHAP",
        "comet_model_name": "xgboost-shap",
        "version": "1.0.1",
        "file_name": "xgboost_SHAP",
    },
    "xgboost-feats-non-corr": {
        "model_type": "xgboost_non_corr",
        "model_desc": "XGBoost with Non Correlated Features",
        "comet_model_name": "xgboost-feats-non-corr",
        "version": "1.0.1",
        "file_name": "xgboost_feats_non_corr",
    },
    "nn-adv": {
        "model_type": "NN_MLP",
        "model_desc": "Neural Network - Advance Features",
        "comet_model_name": "nn-adv",
        "version": "1.0.1",
        "file_name": "NN_adv",
    },
    "lr-all-feats": {
        "model_type": "logreg_all",
        "model_desc": "logistic Regression with all Features in (Q4)",
        "comet_model_name": "lr-all-feats",
        "version": "1.0.0",
        "file_name": "lr_all_feats",
    },
    "lr-non-corr-feats": {
        "model_type": "logreg_non_corr_feats",
        "model_desc": "Logistic Regression without Correlated Features",
        "comet_model_name": "lr-non-corr-feats",
        "version": "1.0.0",
        "file_name": "lr_non_corr_feats",
    },
    "xgboost-SMOTE": {
        "model_type": "xgboost_SMOTE",
        "model_desc": "XGBoost with SMOTE Oversampling",
        "comet_model_name": "xgboost-SMOTE",
        "version": "1.0.0",
        "file_name": "xgboost_SMOTE",
    },
    "lr-SMOTE": {
        "model_type": "logreg_SMOTE",
        "model_desc": "Logistic Regression with SMOTE Oversampling",
        "comet_model_name": "lr-SMOTE",
        "version": "1.0.0",
        "file_name": "lr_SMOTE",
    },
}

## Column and Value Lists
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
    # "prev_event_x_coord",
    "shot_Backhand",
    "shot_Tip-In",
    "shot_Wrist Shot",
    "prev_event_FACEOFF",
    "prev_event_GIVEAWAY",
    "prev_event_HIT",
    "prev_event_SHOT",
]

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

NON_CORR_COLS = [
    "angle",
    "angle_between_prev_event",
    "distance_from_net",
    "distance_from_prev_event",
    "game_sec",
    "is_empty_net",
    "prev_event_BLOCKED_SHOT",
    "prev_event_FACEOFF",
    "prev_event_GIVEAWAY",
    "prev_event_HIT",
    "prev_event_MISSED_SHOT",
    "prev_event_SHOT",
    "prev_event_STOP",
    "prev_event_TAKEAWAY",
    "prev_event_time_diff",
    "rebound_angle",
    "shot_Backhand",
    "shot_Deflected",
    "shot_Slap Shot",
    "shot_Snap Shot",
    "shot_Tip-In",
    "shot_Wrap-around",
    "shot_Wrist Shot",
    "speed",
]

## Fit Scalers
LG_SCALE = {
    "column": ["distance_from_net", "angle"],
    "scale": [23.61912323, 36.63724954],
    "mean": [35.50230422, -0.82357586],
}

NN_SCALE = {
    "column": [
        "game_sec",
        "period",
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
        "shot_Deflected",
        "shot_Slap Shot",
        "shot_Snap Shot",
        "shot_Tip-In",
        "shot_Wrap-around",
        "shot_Wrist Shot",
        "prev_event_BLOCKED_SHOT",
        "prev_event_FACEOFF",
        "prev_event_GIVEAWAY",
        "prev_event_HIT",
        "prev_event_MISSED_SHOT",
        "prev_event_SHOT",
        "prev_event_STOP",
        "prev_event_TAKEAWAY",
    ],
    "scale": [
        1.08239235e03,
        8.78331271e-01,
        2.36487778e01,
        1.90165776e01,
        2.36191232e01,
        3.66372495e01,
        2.90617294e01,
        5.07168419e01,
        1.47170188e01,
        2.13382569e01,
        3.27277378e-01,
        1.67089835e01,
        6.77709669e-02,
        2.74119384e-01,
        1.22086310e-01,
        3.69665556e-01,
        3.54991667e-01,
        2.18226892e-01,
        1.01610543e-01,
        4.99029640e-01,
        3.12633512e-01,
        4.30357912e-01,
        2.91216161e-01,
        3.83772998e-01,
        2.92989537e-01,
        3.93792071e-01,
        1.31522722e-01,
        2.50615455e-01,
    ],
    "mean": [
        1.85135280e03,
        2.05065853e00,
        5.90438612e01,
        -8.61830743e-02,
        3.55023042e01,
        -8.23575860e-01,
        3.89527066e01,
        7.27235426e01,
        1.49760017e01,
        1.17236134e01,
        1.21992701e-01,
        4.78854534e00,
        4.61419475e-03,
        8.18390701e-02,
        1.51341082e-02,
        1.63328978e-01,
        1.47890761e-01,
        5.01366613e-02,
        1.04335616e-02,
        5.31135490e-01,
        1.09794558e-01,
        2.45457139e-01,
        9.35604013e-02,
        1.79503064e-01,
        9.48369076e-02,
        1.91896438e-01,
        1.76082778e-02,
        6.73432150e-02,
    ],
}

LG_ALL_SCALE = {
    "column": [
        "game_sec",
        "period",
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
        "shot_Deflected",
        "shot_Slap Shot",
        "shot_Snap Shot",
        "shot_Tip-In",
        "shot_Wrap-around",
        "shot_Wrist Shot",
        "prev_event_BLOCKED_SHOT",
        "prev_event_FACEOFF",
        "prev_event_GIVEAWAY",
        "prev_event_HIT",
        "prev_event_MISSED_SHOT",
        "prev_event_SHOT",
        "prev_event_STOP",
        "prev_event_TAKEAWAY",
    ],
    "scale": [
        1.07902746e03,
        8.74014465e-01,
        2.36672088e01,
        1.90237928e01,
        2.36322718e01,
        3.66675962e01,
        2.90617294e01,
        5.07168419e01,
        1.47071075e01,
        2.14504660e01,
        3.29407486e-01,
        1.68250122e01,
        6.81434298e-02,
        2.73988261e-01,
        1.21712867e-01,
        3.69455476e-01,
        3.54707655e-01,
        2.18391779e-01,
        1.01768704e-01,
        4.98991774e-01,
        3.14701013e-01,
        4.32541305e-01,
        2.93180794e-01,
        3.86032687e-01,
        2.94962800e-01,
        3.96028122e-01,
        5.23864429e-02,
        2.52348740e-01,
    ],
    "mean": [
        1.84858903e03,
        2.04788404e00,
        5.90430320e01,
        -9.37060485e-02,
        3.55103634e01,
        -8.28603001e-01,
        3.89527066e01,
        7.27235426e01,
        1.51545264e01,
        1.19018606e01,
        1.23847494e-01,
        4.86135099e00,
        4.66529198e-03,
        8.17531435e-02,
        1.50402305e-02,
        1.63098455e-01,
        1.47604655e-01,
        5.02166845e-02,
        1.04664149e-02,
        5.31736563e-01,
        1.11460076e-01,
        2.49185289e-01,
        9.49752823e-02,
        1.82228439e-01,
        9.62711968e-02,
        1.94775940e-01,
        2.75191243e-03,
        6.83518636e-02,
    ],
}

LG_SCALE_df = pd.DataFrame.from_dict(
    LG_SCALE, orient="index", columns=LG_SCALE["column"]
).drop(index="column")
NN_SCALE_df = pd.DataFrame.from_dict(
    NN_SCALE, orient="index", columns=NN_SCALE["column"]
).drop(index="column")
LG_ALL_df = pd.DataFrame.from_dict(
    LG_ALL_SCALE, orient="index", columns=LG_ALL_SCALE["column"]
).drop(index="column")


## Preprocessing Functions
def preprocess_for_model(data, model_type):
    """Preprocess the input data according to the model type"""

    if model_type == "logreg_dist":
        data_processed = LG_preprocess(data, dist=True)

    if model_type == "logreg_ang":
        data_processed = LG_preprocess(data, ang=True)

    if model_type == "logreg_dist_ang":
        data_processed = LG_preprocess(data)

    if model_type == "logreg_all":
        data_processed = preprocess_lr_all(data)

    if model_type == "logreg_SMOTE" or Model_Type == "logreg_non_corr_feats":
        data_processed = preprocess_lr_smote(data)

    if model_type == "xgboost_SHAP":
        data_processed = XGB_SHAP_preprocess(data)

    if model_type == "xgboost_lasso":
        data_processed = XGB_Lasso_preprocess(data)

    if (
        model_type == "xgboost_non_corr"
        or Model_Type == "logreg_SMOTE"
        or Model_Type == "xgboost_SMOTE"
    ):
        data_processed = XGB_Non_Corr_preprocess(data)

    if model_type == "NN_MLP":
        data_processed = NN_preprocess(data)

    return data_processed


def LG_preprocess(data, dist=False, ang=False):

    if dist:
        data = data[["distance_from_net"]]
    elif ang:
        data = data[["angle"]]
    else:
        data = data[["distance_from_net", "angle"]]

    na_mask = data.isnull().any(axis=1)
    data = data[~na_mask]

    if dist:
        data_processed = (data - LG_SCALE_df["distance_from_net"][1]) / LG_SCALE_df[
            "distance_from_net"
        ][0]
    elif ang:
        data_processed = (data - LG_SCALE_df["angle"][1]) / LG_SCALE_df["angle"][0]
    else:
        data_processed = (data - LG_SCALE_df[data.columns].iloc[1]) / LG_SCALE_df[
            data.columns
        ].iloc[0]

    return data_processed


# Logistic Regression
def preprocess_lr_all(data):

    data = data[TRAIN_COLS_PART_4]

    na_mask = data.isnull().any(axis=1)
    data = data[~na_mask]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)

    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    data = pd.get_dummies(data, ["shot", "prev_event"])

    for col in LG_ALL_SCALE.columns:
        if col not in data:
            data[col] = 0

    data = data[LG_ALL_SCALE.columns]
    data_processed = (data - LG_ALL_df[data.columns].iloc[1]) / LG_ALL_df[
        data.columns
    ].iloc[0]

    return data_processed


# Logistic Regression
def preprocess_lr_smote(data):

    data = data[TRAIN_COLS_PART_4]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )
    data = pd.get_dummies(data, ["shot", "prev_event"])

    na_mask = data.isnull().any(axis=1)
    data = data[~na_mask]

    for col in LG_ALL_SCALE.columns:
        if col not in data:
            data[col] = 0

    data = data[LG_ALL_SCALE.columns]
    data_processed = (data - LG_ALL_df[data.columns].iloc[1]) / LG_ALL_df[
        data.columns
    ].iloc[0]

    # the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = data_processed.columns.difference(redundant_feats)
    data_processed = data_processed[selected_feats]

    data_processed = data_processed.fillna(0)

    return data_processed


# XGBoost with SHAP
def XGB_SHAP_preprocess(data):

    data = data[TRAIN_COLS_PART_4]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    data = pd.get_dummies(X_data, ["shot", "prev_event"])

    for col in LG_ALL_SCALE.columns:
        if col not in data:
            data[col] = 0

    data = data[SHAP_COLS]

    return data_processed


# XGBoost with Lasso
def XGB_Lasso_preprocess(data):

    data = data[TRAIN_COLS_PART_4]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    data = pd.get_dummies(data, ["shot", "prev_event"])

    for col in LG_ALL_SCALE.columns:
        if col not in data:
            data[col] = 0

    data_processed = data[LASSO_COLS]

    return data_processed


def XGB_Non_Corr_preprocess(data):

    data = data[TRAIN_COLS_PART_4]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    data = pd.get_dummies(data, ["shot", "prev_event"])

    #    the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = X_data.columns.difference(redundant_feats)
    X_data = X_data[selected_feats]
    X_data = X_data.dropna(axis=0)

    for col in NON_CORR_COLS:
        if col not in X_data:
            X_data[col] = 0
    data_processed = X_data[NON_CORR_COLS]

    return data_processed


# Neural Network with Advance
def NN_preprocess(data):

    data = data[TRAIN_COLS_PART_4]

    data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    data = pd.get_dummies(data, ["shot", "prev_event"])

    data_processed = (data - NN_SCALE_df[data.columns].iloc[1]) / NN_SCALE_df[
        data.columns
    ].iloc[0]

    data_processed = data_processed.fillna(0)

    return data_processed


def flip_coord_to_one_side(game_events_df, right_team, left_team):
    """
    Flip the (x,y) coordinates of the shots events to the right side of the rink for both teams

    :param pd.DataFrame game_events_df: all the game events
    :param str right_team: the team that started the game on the right side of the rink
    :param str left_team: the team that started the game on the left side of the rink
    :return: a dataframe of the game events data after updating
    :rtype: pd.DataFrame
    """
    for idx, row in game_events_df.iterrows():
        period = row["period"]

        # keep the team who started on the right to the right always
        if row["shooter_team_name"] == right_team and period % 2 == 0:
            game_events_df.at[idx, "coordinate_x"] = row["coordinate_x"] * -1
            game_events_df.at[idx, "coordinate_y"] = row["coordinate_y"] * -1

        # flip the team who started on the left to the right always
        elif row["shooter_team_name"] == left_team and period % 2 != 0:
            game_events_df.at[idx, "coordinate_x"] = row["coordinate_x"] * -1
            game_events_df.at[idx, "coordinate_y"] = row["coordinate_y"] * -1
    return game_events_df


def add_milestone2_advanced_metrics(events_df):
    """
    Add advanced features related to the preceding event for each one
    """
    # 1. elapsed time since the game started
    tmp = events_df["time"].str.split(":", expand=True).astype(int)
    events_df["game_sec"] = (events_df["period"] - 1) * 20 * 60 + tmp[0] * 60 + tmp[1]

    # getting information about previous events using df.shift(periods=1)
    prev_events_df = events_df.shift(periods=1)

    # 2.a previous event type
    events_df["prev_event_type"] = prev_events_df["type"]

    # 2.b previous event coordinates
    events_df["prev_event_x_coord"] = prev_events_df["coordinate_x"]
    events_df["prev_event_y_coord"] = prev_events_df["coordinate_y"]

    # 3. time difference in seconds
    events_df["prev_event_time_diff"] = (
        events_df["game_sec"] - prev_events_df["game_sec"]
    )

    # 4. distance between current and previous event
    # first we calcualte the angle between the current and previous events (in degrees)
    events_df["angle_between_prev_event"] = (
        (events_df["angle"] - prev_events_df["angle"]).abs().astype(float).round(4)
    )
    a = events_df["distance_from_net"]
    b = prev_events_df["distance_from_net"]
    # then with the knowledge of the two sides of a triangle and its angle, we can get the third side length
    events_df["distance_from_prev_event"] = np.sqrt(
        a ** 2
        + b ** 2
        - (2 * a * b * np.cos(events_df["angle_between_prev_event"] * np.pi / 180.0))
    )
    events_df["distance_from_prev_event"] = (
        events_df["distance_from_prev_event"].astype(float).round(4)
    )
    # 5. rebound angle is the change in angle between current and previous shot events = [0,180]
    rebound_angle_mask = (
        (events_df["type"] == "SHOT")
        & (events_df["prev_event_type"] == "SHOT")
        & (events_df["shooter_team_name"] == prev_events_df["shooter_team_name"])
        & (events_df["period"] == prev_events_df["period"])
    )
    events_df["rebound_angle"] = events_df["angle_between_prev_event"]
    events_df.loc[~rebound_angle_mask, "rebound_angle"] = 0.0

    # 6. see if the current event is a rebound
    events_df["is_rebound"] = False
    events_df.loc[rebound_angle_mask, "is_rebound"] = True

    # 7. speed of the puck
    speed_mask = events_df["prev_event_time_diff"] > 0
    events_df["speed"] = (
        events_df[speed_mask]["distance_from_prev_event"]
        / events_df[speed_mask]["prev_event_time_diff"]
    )
    events_df["speed"] = events_df["speed"].astype(float).round(4)
    events_df.loc[
        np.isnan(events_df["speed"])
        | (events_df["period"] != prev_events_df["period"]),
        "speed",
    ] = 0.0

    return events_df


def add_milestone2_metrics(events):
    events["distance_from_net"] = (
        (STANDARDIZED_GOAL_COORDINATES[0] - events["coordinate_x"]) ** 2
        + (STANDARDIZED_GOAL_COORDINATES[1] - events["coordinate_y"]) ** 2
    ) ** (0.5)

    events["angle"] = np.arcsin(
        (events["coordinate_y"] / events["distance_from_net"].replace(0, 999)).values
    )  # assumes shots at distance=0 have angle 0

    events["angle"] = (events["angle"] / (np.pi / 2)) * 90  # radians to degrees

    events["is_goal"] = events["type"] == "GOAL"
    events["is_empty_net"] = events["is_empty_net"] == True  # assumes NaN's are False

    return events


def parse_game_data(game_id: str, game_data: dict):
    """
    parse the game data in a json/dictionary format that has all the events information,
    and retrieve the GOAL and SHOT events

    :param str game_id: the unique id of the game
    :param dict game_data: the game data json file as dictionary
    :return: a dataframe of the events information
    :rtype: pd.DataFrame
    """
    events = []
    event_types = set()

    # get the home and away teams
    home_team = game_data["gameData"]["teams"]["home"]["name"]
    away_team = game_data["gameData"]["teams"]["away"]["name"]

    # loop over all events in the game
    for event in game_data["liveData"]["plays"]["allPlays"]:
        # get the event type
        event_result_info = event.get("result", None)
        event_type_id = event_result_info.get("eventTypeId", None)
        # a set for all unique events in the game
        event_types.add(event_type_id)

        event_code = event_result_info.get("eventCode", None)
        event_desc = event_result_info.get("description", None)
        event_secondary_type = event_result_info.get("secondaryType", None)

        # Adding penalty related information
        penaltyMinutes = event_result_info.get("penaltyMinutes", None)

        # event information
        event_about_info = event.get("about", None)
        event_id = event_about_info.get("eventId", None)
        # event index inside the allPlays in the json file
        event_index = event_about_info.get("eventIdx", None)
        period_num = event_about_info.get("period", None)
        period_type = event_about_info.get("periodType", None)
        event_date = event_about_info.get("dateTime", None)
        event_time = event_about_info.get("periodTime", None)
        event_time_remaining = event_about_info.get("periodTimeRemaining", None)
        event_goals_home = event_about_info["goals"]["home"]
        event_goals_away = event_about_info["goals"]["away"]

        # shooting/scoring team information
        shooter_team_info = event.get("team", None)
        shooter_team_id = (
            shooter_team_info.get("id", None) if shooter_team_info else None
        )
        shooter_team_name = (
            shooter_team_info.get("name", None) if shooter_team_info else None
        )
        shooter_team_code = (
            shooter_team_info.get("triCode", None) if shooter_team_info else None
        )

        # players information (i.e. the shooter/scorer and the goalie)
        # Shooter/scorer information
        players_info = event.get("players", None)
        shooter_info = players_info[0].get("player", None) if players_info else None
        shooter_role = players_info[0].get("playerType", None) if players_info else None
        shooter_id = shooter_info.get("id", None) if shooter_info else None
        shooter_name = shooter_info.get("fullName", None) if shooter_info else None

        # Goalie information
        # GOAL event has from 2 to 4 players info: scorer, goalie and up to two assists
        # SHOOT event has 2 players info: shooter and goalie
        # in both cases the goalie is always at the end of the list
        goalie_info = players_info[-1].get("player", None) if players_info else None
        goalie_role = players_info[-1].get("playerType", None) if players_info else None
        goalie_id = goalie_info.get("id", None) if goalie_info else None
        goalie_name = goalie_info.get("fullName", None) if goalie_info else None

        # info specific to GOAL events
        empty_net = None
        game_winning_goal = None
        strength_name = None
        strength_code = None
        empty_net = event_result_info.get("emptyNet", None)
        game_winning_goal = event_result_info.get("gameWinningGoal", None)
        strength_name = (
            event_result_info["strength"]["name"]
            if "strength" in event_result_info.keys()
            else None
        )
        strength_code = (
            event_result_info["strength"]["code"]
            if "strength" in event_result_info.keys()
            else None
        )

        # (x,y) coordinates of the event
        coord_info = event.get("coordinates", None)

        coord_x = coord_info.get("x", None) if coord_info else None
        coord_y = coord_info.get("y", None) if coord_info else None

        event_entry = {
            "id": event_id,
            "event_index": event_index,
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "type": event_type_id,
            "secondary_type": event_secondary_type,
            "description": event_desc,
            "code": event_code,
            "period": period_num,
            "period_type": period_type,
            "time": event_time,
            "time_remaining": event_time_remaining,
            "date": event_date,
            "goals_home": event_goals_home,
            "goals_away": event_goals_away,
            "shooter_team_id": shooter_team_id,
            "shooter_team_name": shooter_team_name,
            "shooter_team_code": shooter_team_code,
            "shooter_name": shooter_name,
            "shooter_id": shooter_id,
            "goalie_name": goalie_name,
            "goalie_id": goalie_id,
            "is_empty_net": empty_net,
            "is_winning_goal": game_winning_goal,
            "strength_name": strength_name,
            "strength_code": strength_code,
            "coordinate_x": coord_x,
            "coordinate_y": coord_y,
        }
        events.append(event_entry)

    events_df = pd.DataFrame(events)

    # calculate the median of the SHOT x_coordinate to see where did the teams start from (left or right)
    if not events_df.empty:
        median_df = (
            events_df[
                ((events_df["period"] == 1) | (events_df["period"] == 3))
                & (events_df["type"] == "SHOT")
            ]
            .groupby(["shooter_team_name", "home_team"])[
                ["coordinate_x", "coordinate_y"]
            ]
            .median()
            .reset_index()
        )
        for idx, row in median_df.iterrows():
            if row["home_team"] == row["shooter_team_name"]:
                if (
                    row["coordinate_x"] > 0
                ):  # means the home team started on the right side
                    events_df = flip_coord_to_one_side(events_df, home_team, away_team)
                else:
                    events_df = flip_coord_to_one_side(events_df, away_team, home_team)

        events_df = add_milestone2_metrics(events_df)
        events_df = add_milestone2_advanced_metrics(events_df)

    return events_df
