""" 
Read and preprocessing input data based on selected model.
"""

import pandas as pd
import sklearn
from utils import INFREQUENT_STOPPAGE_EVENTS, LABEL_COL



# Load data from the specific file path, select all, regular season, or post season data.
def load_dataset(Col, FilePath):

    df = pd.read_csv(FilePath)
    df["game_id"] = df["game_id"].astype(int)

    X_data = df[Col]
    Y_data = df[LABEL_COL].astype(int)

    return X_data, Y_data




def Process_Data(MODELINFO):

    # Retrieve and process data and model
    TestPath = "./data/test_processed.csv"
    X_Test, Y_Test = load_dataset(MODELINFO["Col"], TestPath)    

    if (MODELINFO["model_type"] == "xgboost_non_corr"):
        X_processed = XGB_Non_Corr_preprocess(X_Test)
        return X_processed, Y_Test
    else:
        print("Model does not exist")
        return None



def XGB_Non_Corr_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    #    the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = X_data.columns.difference(redundant_feats)
    X_data = X_data[selected_feats]

    return X_data








