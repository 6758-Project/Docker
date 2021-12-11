"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import sys

sys.path.append("./src")
import os
from pathlib import Path
import logging
from logging import FileHandler
from flask import Flask, jsonify, request, abort
import pandas as pd
import joblib
from utils import retrieve_comet_model

import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    file_handler = FileHandler(LOG_FILE)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(handler)

    app.logger.info("Running intialization routine")

    # TODO: any other initialization before the first request (e.g. load default model)
    global model
    dir_path = "./models"

    if not os.path.exists(dir_path):
        app.logger.info("Creating directory for model storage")
        os.makedirs(dir_path)
        app.logger.warning("No model available")
    else:
        app.logger.info("Model directry exist")
        if os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                app.logger.info("No model in directory")
            else:
                app.logger.info("Loading best model from Milestone 2")
                model = retrieve_comet_model(download=False)
                app.logger.info("Model retrieved")

    return print("Initialization routine complete")


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    # raise NotImplementedError("TODO: implement this endpoint")
    with open(LOG_FILE, "r") as f:
        log_history = f.read()

    response = log_history
    return jsonify(response)  # response must be json serializable!


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
    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.
    # eg: app.logger.info(<LOG STRING>)

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    # Get POST json data
    global model
    comet_model = request.get_json()
    model_name = comet_model["CometModelName"]

    file_path = "./models/xgboost_feats_non_corr.pickle"

    if comet_model["CometModelName"] == "xgboost-feats-non-corr":
        if os.path.isfile(file_path):
            app.logger.info("Model already avaliable")
            model = retrieve_comet_model(download=False)
        else:
            app.logger.info("Downloading")
            model = retrieve_comet_model(comet_model, download=True)
        app.logger.info("Model loading complete")
    else:
        app.logger.info("Invalid Model, default model selected")

    response = "Reistry model retrieved"
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    global model
    json_data = request.get_json(force=True)
    app.logger.info(json_data)

    # TODO:
    # raise NotImplementedError("TODO: implement this enpdoint")

    data = pd.DataFrame(json_data, index=[0])
    response = model.predict(data).tolist()

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/")
def main():
    app.logger.info("Running the Flask App")
    return "The Flask App"
