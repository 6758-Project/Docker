import pandas as pd

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
        incremental_update = (not new_model and not new_game)

        if new_model:
            self.predictor_client.update_model(model_id)
            self.current_model = model_id

        requested_events, previous_returned_idx = \
            self.nhl_api_client.get_game_info(
                game_id=game_id,
            )
        self.curr_game = game_id

        preprocessed_events = get_preprocess_function(model_id)(requested_events)
        preds = self.predictor_client.predict(preprocessed_events)

        if incremental_update:
            new_events = requested_events[requested_events['event_index']>previous_returned_idx]
            new_events = pd.concat([requested_events, preds], axis=1)
            self.events = pd.concat([self.events, new_events])  # append
        else:
            self.events = pd.concat([requested_events, preds], axis=1)  # overwrite

        self.render_dashboard()  # TODO should this be a return?

    def render_dashboard(self):
        """ Generates expected dashboard layout """
        # TODO -- will primarily use self.events