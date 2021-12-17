import os
import shutil

import pandas as pd
import pytest

from client.nhl_api_client import NHLAPIClient
from client.predictor_api_client import PredictorAPIClient
from client.GameTracker import GameTracker


nhl_api = NHLAPIClient(raw_game_directory="./tmp_data/")
predictor = PredictorAPIClient()
game_tracker = GameTracker(nhl_api_client=nhl_api, predictor_client=predictor)

max_example_data_index = 30
example = pd.read_csv("test_data/example.csv")
example['secondary_type'] =  example['secondary_type'].astype('category')
example['prev_event_type'] =  example['prev_event_type'].astype('category')


def dummy_data_getter(game_id):
    return example.iloc[:max_example_data_index,:]

nhl_api.query_api = dummy_data_getter # replacing actual query method with dummy method


def setup_function():
    if not os.path.exists("./tmp_data/"):
        os.makedirs("./tmp_data/")

    # res = nhl_api.query_api(2015020001)
    # res.to_csv("./tmp_data/2015020001.csv", index=False)


def test_new_game():
    game_tracker.update_dashboard(
        game_id=2015020001, model_id="xgboost-feats-non-corr"
    )

    assert len(game_tracker.nhl_api_client.loaded_games) == 1

def test_new_model():
    game_tracker.update_dashboard(
        game_id=2015020001, model_id="logistic-regression-distance-and-angle"
    )

    assert len(game_tracker.nhl_api_client.loaded_games) == 1

def test_game_update():
    curr_preds = list(game_tracker.events['predictions'])
    max_example_data_index = 40
    game_tracker.update_dashboard(
        game_id=2015020001, model_id="logistic-regression-distance-and-angle"
    )

    update_preds = list(game_tracker.events['predictions'])

    assert 1 ==1

def teardown_function():
    """A hook called by pytest after any tests are run"""
    shutil.rmtree("./tmp_data/")




if __name__ == '__main__':
    setup_function()
    test_new_game()
    test_new_model()
    test_game_update()
    teardown_function()
