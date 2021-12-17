""" Tests for the Game Tracker

If actively developing, use like a normal python script (i.e. calling the main func)
Otherwise, call via pytest:  python -m pytest test_game_events.py
"""
import math
import os
import shutil

import pandas as pd
import pytest

from client.nhl_api_client import NHLAPIClient, UnknownGameException
from client.predictor_api_client import PredictorAPIClient, UnknownModelException
from client.GameTracker import GameTracker


nhl_api = NHLAPIClient(raw_game_directory="./tmp_data/")
predictor = PredictorAPIClient()
game_tracker = GameTracker(nhl_api_client=nhl_api, predictor_client=predictor)

max_example_data_index = 30
example = pd.read_csv("test_data/example.csv")
example['secondary_type'] =  example['secondary_type'].astype('category')
example['prev_event_type'] =  example['prev_event_type'].astype('category')
example = example.drop(columns=["Unnamed: 0"], errors='ignore')


def dummy_data_getter(game_id):
    if not isinstance(game_id, int):
        raise UnknownGameException(str(game_id))
    else:
        return example.iloc[:max_example_data_index,:]

nhl_api.query_api = dummy_data_getter # replacing actual query method with dummy method


def setup_function():
    if not os.path.exists("./tmp_data/"):
        os.makedirs("./tmp_data/")

def test_new_game():
    game_tracker.update(
        game_id=2015020001, model_id="xgboost-feats-non-corr"
    )

    assert len(game_tracker.nhl_api_client.loaded_games) == 1

def test_new_model():
    game_tracker.update(
        game_id=2015020001, model_id="logistic-regression-distance-and-angle"
    )

    assert len(game_tracker.nhl_api_client.loaded_games) == 1

def test_game_update():
    curr_preds = list(game_tracker.events['predictions'])

    global max_example_data_index
    max_example_data_index = 40  # 10 new events in dummy example

    game_tracker.update(
        game_id=2015020001, model_id="logistic-regression-distance-and-angle"
    )
    update_preds = list(game_tracker.events['predictions'])
    assert len(update_preds) - len(curr_preds) == 10
    assert all([cp == up or (math.isnan(cp) and math.isnan(up)) for cp, up in zip(curr_preds, update_preds)])

def test_new_game_2():
    game_tracker.update(
        game_id=2015020002, model_id="logistic-regression-distance-and-angle"
    )

    assert len(game_tracker.nhl_api_client.loaded_games) == 2

def test_bad_game_id():
    with pytest.raises(UnknownGameException):
        game_tracker.update(
            game_id='xyz', model_id="logistic-regression-distance-and-angle"
        )

def test_bad_model_id():
    with pytest.raises(UnknownModelException):
        game_tracker.update(
        game_id=2015020002, model_id="xyz"
        )

def teardown_function():
    """A hook called by pytest after any tests are run"""
    shutil.rmtree("./tmp_data/")




if __name__ == '__main__':
    setup_function()
    test_new_game()
    test_new_model()
    test_game_update()
    test_new_game_2()
    test_bad_game_id()
    test_bad_model_id()
    teardown_function()
