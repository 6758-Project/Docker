"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import logging
import pickle
import json
from flask import Flask, jsonify, request
from comet_ml import API
import pandas as pd

import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

LABEL_COL = "is_goal"

MODEL_DIR = "./models"
MODEL_NAME = "xgboost_feats_non_corr.pickle"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

global comet_model


app = Flask(__name__)


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
    logging.basicConfig(filename=LOG_FILE, format=logging_format, level=logging.INFO)

    # any other initialization before the first request (e.g. load default model)
    try:
        with open(MODEL_PATH, "rb") as model_pickle_file:
            comet_model = pickle.load(model_pickle_file)
        app.logger.info("default model was found and loaded successfully ...")
    except:
        app.logger.exception(f"default model failed to load from: {MODEL_PATH}")

    pass


@app.route("/", methods=["GET"])
def home():
    """home screen"""
    app.logger.info("welcome screen")
    return "<h1>Hello Milestone 3!</h1>"


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # # read the log file specified and return the data

    # html friendly format
    # logs_lines = "<h3>"
    # with open("flask.log", "r") as f:
    #     lines = f.read().splitlines()
    #     for line in lines:
    #         logs_lines += line.strip()
    #         logs_lines += "<br>"
    # logs_lines += "</h3>"
    # response = logs_lines

    # json friendly format
    logs_lines = {}
    with open("flask.log", "r") as f:
        lines = f.read().splitlines()
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
    COMET_MODEL_INFO = json.loads(json_req)
    output_msg = None

    # check to see if the model you are querying for is already downloaded
    # if yes, load that model and write to the log about the model change.
    # eg: app.logger.info(<LOG STRING>)
    if os.path.isfile(MODEL_PATH):
        app.logger.info(f"model already downloaded in {MODEL_PATH}")

        try:
            with open(MODEL_PATH, "rb") as model_pickle_file:
                comet_model = pickle.load(model_pickle_file)
            app.logger.info("model loaded successfully ...")
            output_msg = "Model loaded successfully - for more information check \\logs"
        except:
            output_msg = "Model failed to load - for more information check \\logs"
            app.logger.exception(f"model failed to load from {MODEL_PATH}")

    # if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the
    # currently loaded model

    try:
        # make sure the path exists
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        comet_api = API()
        comet_api.download_registry_model(
            COMET_MODEL_INFO["workspace"],
            COMET_MODEL_INFO["model_name"],
            COMET_MODEL_INFO["version"],
            output_path=MODEL_DIR,
            expand=True,
        )
        app.logger.info(f"model downloaded successfully in: {MODEL_PATH}")

        try:
            with open(MODEL_PATH, "rb") as model_pickle_file:
                comet_model = pickle.load(model_pickle_file)
            output_msg = "model downloaded and loaded successfully - for more information check \\logs"
            app.logger.info("model loaded successfully ...")

        except:
            output_msg = f"model failed to load from: {MODEL_PATH}"
            app.logger.exception(f"model failed to load from: {MODEL_PATH}")

    except:
        output_msg = "model failed to download - for more information check \\logs"
        app.logger.exception("model failed to download")

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    response = output_msg

    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_req = request.get_json()

    # predict the datapoint (first row if a dataframe has rows > 1)
    data_dct = json.loads(json_req)
    data_df = pd.DataFrame(data_dct)

    # remove the label column if it exists
    data_df = data_df[data_df.columns.difference([LABEL_COL])]

    app.logger.info(f"Predicting: {len(data_df)} events")

    # get the prediction and prediction probability
    try:

        with open(MODEL_PATH, "rb") as model_pickle_file:
            comet_model = pickle.load(model_pickle_file)

        y_pred = comet_model.predict(data_df)
        y_proba = comet_model.predict_proba(data_df)[:, 1]

        app.logger.info(f"Prediction of {LABEL_COL}?: {y_pred}")
        app.logger.info(f"Probability of the prediction: {y_proba}")
        
        pred_df = pd.DataFrame(y_pred, columns=['predictions']) 
        response = pred_df.to_json()
        app.logger.info("Predictions and their probabilities retrieved successfully ... ")

    except:
        app.logger.exception(f"model failed to load from: {MODEL_PATH}")
        app.logger.exception(f"No model to use. comet_model={comet_model}")
        response = None

    return jsonify(response)  # response must be json serializable!
