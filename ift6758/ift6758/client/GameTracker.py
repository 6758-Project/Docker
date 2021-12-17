import pandas as pd

from .nhl_api_client import UnknownGameException
from .predictor_api_client import UnknownModelException
from .utils import get_preprocess_function

class GameTracker:
    """ Generates a real-time NHL game tracker dashboard"""

    def __init__(self, nhl_api_client, predictor_client):
        self.nhl_api_client = nhl_api_client
        self.predictor_client = predictor_client

        self.curr_game = None
        self.current_model = None
        self.events = None

    def update_dashboard(self, game_id, model_id):
        new_model = (self.current_model != model_id)
        new_game = (self.curr_game != game_id)

        try:
            game_events = self.nhl_api_client.get_game_info(game_id=game_id)
            self.curr_game = game_id
        except UnknownGameException as uge:
            err_msg = f"Game ID {game_id} not recognized, so dashboard unchanged"
            return self.render_dashboard(err_msg=err_msg)

        err_msg = None
        if new_model:
            try:
                self.predictor_client.update_model(model_id)
                self.current_model = model_id
            except UnknownModelException as ume:
                err_msg = f"Model unchanged because " + ume.message
                new_model = False

        if new_model or new_game:
            self.events = pd.DataFrame(columns=self.events.columns)  # resets to empty dataframe

        preprocessed_events = get_preprocess_function(model_id)(game_events)

        new_pred_inputs = preprocessed_events.iloc[(len(self.events)+1):]
        new_preds = self.predictor_client.predict(new_pred_inputs)

        new_events = pd.concat([game_events.iloc[(len(self.events)+1):], new_preds], axis=1)
        self.events = pd.concat([self.events, new_events])

        return self.render_dashboard(err_msg=err_msg)


    def render_dashboard(self, err_msg: str = None):
        """ Generates expected dashboard layout

        Args:
          * err_msg  if not None, prints over dashboard
        """
        pass # TODO -- will primarily use self.events
