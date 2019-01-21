import os
import random
from abc import abstractmethod

import tensorflow as tf
import numpy as np

import settings


class Dataset(object):
    def __init__(self, name, folders, parse_function):
        self.name = name
        self.folders = folders
        self.base_address = folders[0][:folders[0].rindex('/')]
        self.class_folders = {folder_name[folder_name.rindex('/') + 1:]: folder_name for folder_name in folders}
        self.parse_function = parse_function

    def get_supervised_meta_learning_tasks(self, meta_batch_size, n, k):
        def per_directory_dataset(directory_glob):
            return tf.data.Dataset.list_files(directory_glob, shuffle=True)

        classes = [class_name + '/*' for class_name in self.folders]

        datasets = tf.data.Dataset.from_tensor_slices(classes)
        datasets = datasets.shuffle(len(classes))
        dataset = datasets.interleave(per_directory_dataset, cycle_length=len(classes), block_length=2 * k)
        dataset = dataset.take(n * meta_batch_size * 2 * k)

        dataset = dataset.map(self.parse_function)

        dataset = dataset.batch(2 * k)
        dataset = dataset.batch(n)
        dataset = dataset.batch(meta_batch_size)

        dataset = dataset.repeat(-1)

        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()

        train_task = data[:, :, :k, ...]
        val_task = data[:, :, k:, ...]

        train_task = tf.reshape(train_task, (meta_batch_size, n * k, -1))
        val_task = tf.reshape(val_task, (meta_batch_size, n * k, -1))

        labels = []
        for i in range(n):
            labels.append([i] * 2 * k)

        labels = np.array(labels).reshape(2 * k * n)
        labels = tf.stack((labels,) * meta_batch_size)

        train_labels = labels[:, ::2, ...]
        val_labels = labels[:, 1::2, ...]

        train_labels = tf.one_hot(train_labels, depth=n)
        val_labels = tf.one_hot(val_labels, depth=n)

        return train_task, val_task, train_labels, val_labels

    def get_all_class_instances(self, class_name, batch_size=5):
        instances = os.listdir(self.class_folders[class_name])
        instances = [os.path.join(self.base_address, class_name, instance_address) for instance_address in instances]
        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.parse_function)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        images = iterator.get_next()
        return images

    def get_iterator(self):
        pass

    def __str__(self):
        return self.name


class AbstractDataset(object):
    def __init__(self, parse_function=None):
        self.name = self.get_name()
        self.train_folders, self.val_folders, self.test_folders = self.define_train_val_test_folders()
        self.parse_function = self.define_parse_function(parse_function)

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def define_train_val_test_folders(self):
        pass

    @abstractmethod
    def default_parse_function(self):
        pass

    def define_parse_function(self, parse_function):
        if parse_function is not None:
            return parse_function
        return self.default_parse_function()

    def get_train_dataset(self):
        return Dataset(self.name + '-train', self.train_folders, self.parse_function)

    def get_validation_dataset(self):
        return Dataset(self.name + '-val', self.val_folders, self.parse_function)

    def get_test_dataset(self):
        return Dataset(self.name + '-test', self.test_folders, self.parse_function)


class MiniImagenetDataset(AbstractDataset):
    def define_train_val_test_folders(self):
        train_address = os.path.join(settings.MINIIMAGENET_PROCESSED_DATA_ADDRESS, 'train')
        train_folders = [os.path.join(train_address, class_name) for class_name in os.listdir(train_address)]

        val_address = os.path.join(settings.MINIIMAGENET_PROCESSED_DATA_ADDRESS, 'val')
        val_folders = [os.path.join(val_address, class_name) for class_name in os.listdir(val_address)]

        test_address = os.path.join(settings.MINIIMAGENET_PROCESSED_DATA_ADDRESS, 'test')
        test_folders = [os.path.join(test_address, class_name) for class_name in os.listdir(test_address)]
        return train_folders, val_folders, test_folders

    def default_parse_function(self):
        def _parse_function(example_address):
            image = tf.image.decode_jpeg(tf.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_images(image, (84, 84))
            return (image / 255.) * 2 - 1

        return _parse_function

    def get_name(self):
        return 'miniimagenet'


class OmniglotDataset(AbstractDataset):
    def define_train_val_test_folders(self):
        folders = os.listdir(settings.OMNIGLOT_PROCESSED_DATA_ADDRESS)
        random.seed(settings.PYTHON_RANDOM_SEED)
        random.shuffle(folders)

        train_folders = [
            os.path.join(settings.OMNIGLOT_PROCESSED_DATA_ADDRESS, folder_name) for folder_name in folders[:1200]
        ]
        validation_folders = [
            os.path.join(settings.OMNIGLOT_PROCESSED_DATA_ADDRESS, folder_name) for folder_name in folders[1200:1300]
        ]
        test_folders = [
            os.path.join(settings.OMNIGLOT_PROCESSED_DATA_ADDRESS, folder_name) for folder_name in folders[1300:]
        ]
        return train_folders, validation_folders, test_folders

    def default_parse_function(self):
        def _parse_function(example_address):
            image = tf.image.decode_jpeg(tf.read_file(example_address))
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_images(image, (28, 28))
            image = 1 - (image / 255.)
            return image

        return _parse_function

    def get_name(self):
        return 'omniglot'


class AirCraftDataset(AbstractDataset):
    def define_train_val_test_folders(self):
        folders = os.listdir(settings.PROCESSED_AIRCRAFT_ADDRESS)
        random.seed(settings.PYTHON_RANDOM_SEED)
        random.shuffle(folders)

        train_folders = [
            os.path.join(settings.PROCESSED_AIRCRAFT_ADDRESS, folder_name) for folder_name in folders[:64]
        ]
        validation_folders = [
            os.path.join(settings.PROCESSED_AIRCRAFT_ADDRESS, folder_name) for folder_name in folders[64:80]
        ]
        test_folders = [
            os.path.join(settings.PROCESSED_AIRCRAFT_ADDRESS, folder_name) for folder_name in folders[80:]
        ]
        return train_folders, validation_folders, test_folders

    def default_parse_function(self):
        def _parse_function(example_address):
            image = tf.image.decode_jpeg(tf.read_file(example_address))
            image = tf.cast(image, tf.float32)

            # Remove the banner from the bottom
            image = image[:-20, ...]

            # Pad the smaller size with zeros
            new_size = tf.maximum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.pad_to_bounding_box(
                image,
                tf.cast((new_size - tf.shape(image)[0]) / 2, tf.int32),
                tf.cast((new_size - tf.shape(image)[1]) / 2, tf.int32),
                new_size,
                new_size,
            )

            # Resize to 84 x 84
            image = tf.image.resize_images(image, (84, 84), method=tf.image.ResizeMethod.BILINEAR)

            return image

        return _parse_function

    def get_name(self):
        return 'aircraft'


if __name__ == '__main__':
    aircraft_dataset = AirCraftDataset()
    val_dataset = aircraft_dataset.get_validation_dataset()
    train_task, val_task, train_labels, val_labels = val_dataset.get_supervised_meta_learning_tasks(
        meta_batch_size=3,
        n=6,
        k=2
    )

    with tf.Session() as sess:
        for experiment in range(1):
            if experiment % 100 == 0:
                print(experiment)

            tr_np, val_np, tr_lb_np, val_lb_np = sess.run(
                (train_task, val_task, train_labels, val_labels)
            )

            print(tr_np.shape)
            print(val_np)
            print(tr_lb_np)
            print(val_lb_np)
            import matplotlib.pyplot as plt

            plt.imshow(tr_np[0, 0, ...].reshape(256, 256, 3) / 255.)
            # plt.imshow(tr_np[0, 0, ...].reshape(84, 84, 3))
            plt.show()
