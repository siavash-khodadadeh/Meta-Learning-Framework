import os
import shutil

import settings


def prepare_omniglot_dataset(omniglot_dataset_address, processed_dataset_address):
    for alphabet in os.listdir(omniglot_dataset_address):
        alphabet_address = os.path.join(omniglot_dataset_address, alphabet)
        if os.path.isdir(alphabet_address):
            for character in os.listdir(alphabet_address):
                character_address = os.path.join(alphabet_address, character)
                target_address = os.path.join(processed_dataset_address, alphabet + '_' + character)
                print(character_address)
                print(target_address)

                if not os.path.exists(target_address):
                    shutil.copytree(character_address, target_address)


def prepare_aircraft_dataset():
    for partition in ('train', 'test', 'val'):
        partition_address = os.path.join(settings.ARICRAFT_DATA_ADDRESS, 'images_variant_' + partition + '.txt')
        with open(partition_address) as f:
            for line in f:
                data_point, class_name = line.split(' ', 1)
                data_point += '.jpg'
                class_name = class_name[:-1]
                class_name = class_name.replace('/', '-')

                source_address = os.path.join(settings.ARICRAFT_DATA_ADDRESS, 'images', data_point)
                target_directory = os.path.join(settings.PROCESSED_AIRCRAFT_ADDRESS, class_name)
                if not os.path.exists(target_directory):
                    os.mkdir(target_directory)

                target_address = os.path.join(target_directory, data_point)

                shutil.copyfile(source_address, target_address)
