import os


PROJECT_ROOT = os.path.dirname(__file__)
PROCESSED_DATASET_ADDRESS = os.path.join(PROJECT_ROOT, 'datasets')

PYTHON_RANDOM_SEED = 4

#  Raw data for omniglot
OMNIGLOT_DATA_ADDRESS = '/home/siavash/meta-dataset/omniglot_resized/'

#  Processed omniglot data
OMNIGLOT_PROCESSED_DATA_ADDRESS = os.path.join(PROCESSED_DATASET_ADDRESS, 'omniglot')


#  Raw data for aircraft
ARICRAFT_DATA_ADDRESS = '/home/siavash/meta-dataset/aircraft/'

#  Processed aircraft data
PROCESSED_AIRCRAFT_ADDRESS = os.path.join(PROCESSED_DATASET_ADDRESS, 'aircraft')

#  Processed mini-imagenet data
MINIIMAGENET_PROCESSED_DATA_ADDRESS = '/home/siavash/meta-dataset/miniImagenet/'

DEFAULT_LOG_DIR = './logs/'
