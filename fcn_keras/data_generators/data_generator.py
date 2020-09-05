
import numpy as np

from fcn_keras.base.base_data_generator import BaseDataGenerator
from fcn_keras.data_generators.data_augmentation import data_aug_functions
from fcn_keras.preprocessing.preproc_functions import read_image, \
    read_annotation, convert_annotation_one_hot
from keras.applications.vgg16 import preprocess_input


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, dataset, shuffle=True,
                 use_data_augmentation=False):
        super().__init__(config, shuffle, use_data_augmentation)
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.indices = np.arange(self.dataset_len)
        self.num_classes = self.config['network']['num_classes']
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:
                               (index + 1) * self.batch_size]

        # Find list of IDs
        dataset_temp = [self.dataset[k] for k in indices]

        # Generate data
        x, y = self.data_generation(dataset_temp)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch 
        """

        if self.shuffle:
            np.random.shuffle(self.indices)

    def data_generation(self, dataset_temp):

        batch_x = []
        batch_y = []

        for elem in dataset_temp:
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']

            image = read_image(self.dataset_folder, elem['filename'], y_size,
                               x_size)
            annotation = read_annotation(self.dataset_folder,
                                         elem['annotation'], y_size, x_size)

            if self.use_data_aug:
                image, annotation = data_aug_functions(image, annotation,
                                                       self.config)

            annotation_one_hot = convert_annotation_one_hot(annotation,
                                                            self.num_classes)
            image = preprocess_input(image, mode='tf')

            batch_x.append(image)
            batch_y.append(annotation_one_hot)

        batch_x = np.asarray(batch_x, dtype=np.float32)
        batch_y = np.asarray(batch_y, dtype=np.float32)

        return batch_x, batch_y

    def get_full_dataset(self):

        y_size = self.config['image']['image_size']['y_size']
        x_size = self.config['image']['image_size']['x_size']

        images = []
        labels = []
        for elem in self.dataset:
            filename = elem['filename']
            images.append(read_image(self.dataset_folder, filename, y_size,
                                     x_size))
            labels.append(read_annotation(self.dataset_folder,
                                          elem['annotation'], y_size, x_size))
        images = np.array(images)
        images = preprocess_input(images, mode='caffe')
        labels_one_hot = convert_annotation_one_hot(np.array(labels),
                                                    self.num_classes)

        return images, labels_one_hot
