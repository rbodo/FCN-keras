from keras.models import model_from_json

from fcn_keras.base.base_predictor import BasePredictor


class PredictorFCN(BasePredictor):

    def __init__(self, config, model=None):
        """
        Constructor
        """
        super().__init__(config)
        self.graph_path = self.config['network']['graph_path']
        self.weights_path = self.config['predict']['weights_file']

        self.batch_size = self.config['predict']['batch_size']
        self.model = self.load_model() if model is None else model

    def load_model(self, **kwargs):
        json_file = open(self.graph_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.weights_path)

        return model

    def predict(self, images):
        return self.model.predict(images, batch_size=self.batch_size)
