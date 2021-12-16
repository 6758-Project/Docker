import os
import shutil

import pytest

from client.nhl_api_client import NHLAPIClient
from client.predictor_api_client import PredictorAPIClient
from client.GameTracker import GameTracker


nhl_api = NHLAPIClient(raw_game_directory="./tmp_data/")
predictor = PredictorAPIClient()
game_tracker = GameTracker(nhl_api_client=nhl_api, predictor_client=predictor)

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

def teardown_function():
    """A hook called by pytest after any tests are run"""
    shutil.rmtree("./tmp_data/")




if __name__ == '__main__':
    setup_function()
    test_new_game()
    test_new_model()
    teardown_function()
