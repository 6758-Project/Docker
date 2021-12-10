""" Utilities for training models """

from comet_ml import API
import logging
import pickle
import warnings
from comet_ml import Experiment
from comet_ml import ConfusionMatrix
import sklearn
import os

logging.basicConfig(level=logging.INFO)


# Defined Constants

EXP_KWARGS = {
    "project_name": "ift6758-hockey",
    "workspace": "tim-k-lee",
    "auto_param_logging": True,
}

LABEL_COL = "is_goal"

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
]  # 'period_type', 'shooter_team_name', 'shooter_id', 'goalie_name',


MODEL_XGB_NC = {
    "model_type": "xgboost_non_corr",
    "Name": "XGBoost with Non Correlated Features",
    "CometModelName": "xgboost-feats-non-corr",
    "Version": "1.0.1",
    "FileName": "xgboost_feats_non_corr",
    "Col": TRAIN_COLS_PART_4,
}




RANDOM_STATE = 1729


# Functions
        
# Download registered models from Comet and load the models
def Retrieve_Comet(ModelComet, Download = False):
    if Download == True:
        api = API()
        api.download_registry_model(
            EXP_KWARGS["workspace"],
            ModelComet["CometModelName"],
            ModelComet["Version"],
            output_path="./models",
            expand=True,
        )
    else:
        with open(
            os.path.join("./models/", ModelComet["FileName"] + ".pickle"), "rb"
        ) as fid:
            Model = pickle.load(fid)
    return Model
