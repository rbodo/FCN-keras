from keras.models import Model, model_from_json

from fcn_keras.base.base_model import BaseModel
from fcn_keras.losses.custom_losses import custom_categorical_crossentropy
from fcn_keras.models.encoder import encoder_graph, encoder_graph_vgg16
from fcn_keras.models.decoder import decoder_graph_8x, decoder_graph_16x, \
    decoder_graph_32x


class ModelFCN(BaseModel):

    def __init__(self, config):
        """Constructor"""
        super().__init__(config)
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.num_classes = self.config['network']['num_classes']
        self.use_pretrained_weights = self.config['train'][
            'weights_initialization']['use_pretrained_weights']
        self.train_from_scratch = self.config['network']['train_from_scratch']
        self.graph_path = self.config['network']['graph_path']
        self.decoder = self.config['network']['decoder']
        self.model = self.build_model()

    def build_model(self):

        model = self.build_graph()
        model.compile(self.optimizer, custom_categorical_crossentropy())
        model.summary()

        return model

    def build_graph(self):

        if self.use_pretrained_weights:
            json_file = open(self.graph_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()

            return model_from_json(loaded_model_json)

        # Initialize encoder randomly or with vgg16 weights.
        encoder = encoder_graph if self.train_from_scratch \
            else encoder_graph_vgg16
        input_graph, pool_3, pool_4, encoder_out = encoder(
            self.y_size, self.x_size, self.num_channels, self.num_classes)

        if self.decoder == 'decoder_8x':
            decoder_out = decoder_graph_8x(pool_3, pool_4, encoder_out,
                                           self.num_classes)
        elif self.decoder == 'decoder_16x':
            decoder_out = decoder_graph_16x(pool_4, encoder_out,
                                            self.num_classes)
        elif self.decoder == 'decoder_32x':
            decoder_out = decoder_graph_32x(encoder_out, self.num_classes)
        else:
            raise Exception("Unknown decoder")

        return Model(input_graph, decoder_out)
