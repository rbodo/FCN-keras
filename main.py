import argparse
import os
import time

import numpy as np
import json
import yaml
from PIL import Image

from fcn_keras.data_generators.data_generator import DataGenerator
from fcn_keras.models.model import ModelFCN
from fcn_keras.trainers.trainer import TrainerFCN
from fcn_keras.utils.score_prediction import score_prediction
from fcn_keras.preprocessing.preproc_functions import read_image
from keras.applications.vgg16 import preprocess_input


def train(args):
    """Train a model on the train set defined in labels.json."""

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    train_generator = DataGenerator(config, dataset['train'], shuffle=True,
                                    use_data_augmentation=config['data_aug'][
                                        'use_data_aug'])

    val_generator = DataGenerator(config, dataset['val'], shuffle=True,
                                  use_data_augmentation=False)

    train_model = ModelFCN(config)
    trainer = TrainerFCN(config, train_model, train_generator, val_generator)

    trainer.train()


def predict_on_test(args):
    """Predict on the test set defined in labels.json."""

    with open(args.conf) as f:
        config = yaml.full_load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    generator = DataGenerator(config, dataset['test'], shuffle=False,
                              use_data_augmentation=False)

    print("Loading data...")
    images, labels = generator.get_full_dataset()

    print("Predicting...")
    pred = args.model.predict(images, config['predict']['batch_size'],
                              verbose=1)

    print("Evaluating results...")
    pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU, pred_ = \
        score_prediction(labels, pred, config['network']['num_classes'])

    print("Total fraction of correct pixels: {:.2%}".format(pixel_accuracy))
    print("Correct pixels averaged over classes: {:.2%}".format(mean_accuracy))
    print("Mean IoU: {:.2f}".format(mean_IoU))
    print("Freq-weighted mean IoU: {:.2f}".format(freq_weighted_mean_IoU))

    ts = time.time()
    path = os.path.abspath('./out/{}'.format(ts))
    os.makedirs(path)
    print("Saving annotated images to {}...".format(path))
    masks = np.multiply(pred_, [[0, 255, 0, 127]]).astype(np.uint8)
    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    for mask, d in zip(masks, generator.dataset):
        filename = d['filename']
        image = read_image(generator.dataset_folder, filename, y_size, x_size)
        mask = Image.fromarray(mask, mode='RGBA')
        img = Image.fromarray(image)
        img.paste(mask, box=None, mask=mask)
        img.save(os.path.join(path, os.path.basename(filename)))


def predict(args):
    """Predict on a single image."""

    with open(args.conf) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']

    image = read_image('./', args.filename, y_size, x_size)
    image = preprocess_input(image, mode='tf')

    prediction = args.model.predict(np.expand_dims(image, axis=0))[0]
    pred_classes = np.argmax(prediction, axis=-1)

    print("pred_classes:", pred_classes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file',
                        required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')
    group.add_argument('--predict_on_test', action='store_true',
                       help='Predict on test set')
    group.add_argument('--predict', action='store_true',
                       help='Predict on single file')

    parser.add_argument('--filename', help='path to file')

    _args = parser.parse_args()

    if _args.predict_on_test:
        print('Predicting on test set')
        predict_on_test(_args)

    elif _args.predict:
        if _args.filename is None:
            raise Exception('missing --filename FILENAME')
        else:
            print('predict')
        predict(_args)

    elif _args.train:
        print('Starting training')
        train(_args)
    else:
        raise Exception('Unknown args')
