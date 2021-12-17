"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import json
import logging
import os
import pickle
import re
from time import gmtime, strftime

from flask import Flask, jsonify, request
from comet_ml import API

import pandas as pd
import numpy as np

import ift6758

default_flask_file = re.sub("[^0-9a-zA-Z]+", "_", strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
LOG_FILE_PATH = os.environ.get("FLASK_LOG", f"flask_{default_flask_file}.log")

LABEL_COL = "is_goal"

MODEL_DIR = "./models"

DEFAULT_MODEL_INFO = {
    "project_name": "ift6758-hockey",
    "workspace": "tim-k-lee",
    "model_type": "xgboost_non_corr",
    "model_desc": "XGBoost with Non Correlated Features",
    "comet_model_name": "xgboost-feats-non-corr",
    "version": "1.0.1",
    "file_name": "xgboost_feats_non_corr",
}

comet_model = None

app = Flask(__name__)


def load_model(model_info_dct):
    """loads a model from Comet API"""

    global comet_model

    model_path = os.path.join(MODEL_DIR, model_info_dct["file_name"] + ".pickle")

    if os.path.isfile(model_path):
        app.logger.info(f"found {model_info_dct['file_name']} model in {model_path}")

    else:
        comet_api = API()
        comet_api.download_registry_model(
            model_info_dct["workspace"],
            model_info_dct["comet_model_name"],
            model_info_dct["version"],
            output_path=MODEL_DIR,
            expand=True,
        )
        app.logger.info(f"downloaded [{model_info_dct['file_name']}]")

    with open(model_path, "rb") as model_pickle_file:
       comet_model = pickle.load(model_pickle_file)
       app.logger.info(f"loaded {model_info_dct['file_name']} model")


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """

    # setup basic logging configuration
    logging_format = (
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
    )
    logging.basicConfig(
        filename=LOG_FILE_PATH, format=logging_format, level=logging.INFO
    )

    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("")

    app.logger.info(f"prediction API is now running")

    load_model(DEFAULT_MODEL_INFO)


@app.route("/", methods=["GET"])
def home():
    """home screen"""
    app.logger.info("welcome screen")
    return "<h1>Hello Milestone 3!</h1>"


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # read the log file specified and return the data
    logs_lines = {}
    with open(LOG_FILE_PATH, "r") as log_file:
        lines = log_file.read().splitlines()
        for idx, line in enumerate(lines):
            logs_lines[f"log_line#{idx}"] = line.strip()
    response = json.dumps(logs_lines)

    return response  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data (the data point to predict)
    json_req = request.get_json()
    app.logger.info(f"Model information: {json_req}")
    comet_model_info = json.loads(json_req)

    try:
        load_model(comet_model_info)
        output_msg = f"{comet_model_info['file_name']} model loaded successfully ..."
    except:
        output_msg = f"{comet_model_info['file_name']} model failed to be loaded"
        app.logger.exception(f"{comet_model_info['file_name']} model failed to load")

    response = output_msg

    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    requested_preds = request.get_json()
    requested_preds = pd.DataFrame(json.loads(requested_preds))

    if len(requested_preds) == 0:
        return jsonify(pd.DataFrame(columns=["predictions"]).to_json())  # empty df

    global comet_model

    # remove the label column if it exists
    requested_preds = requested_preds[requested_preds.columns.difference([LABEL_COL])]

    app.logger.info(f"Predicting: {len(requested_preds)} events")

    try:
        y_proba = comet_model.predict_proba(requested_preds)[:, 1]
        pred_df = pd.DataFrame(y_proba, columns=["predictions"])
        response = pred_df.to_json()
        app.logger.info("Predictions retrieved successfully ... ")

    except:
        app.logger.exception(f"model failed to predict the {len(requested_preds)} events")
        response = None

    return jsonify(response)  # response must be json serializable!
