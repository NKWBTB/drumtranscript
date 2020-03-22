# Settings for tensorflow datasets
TFDS_NAME = "groove/full-16000hz"
TFDS_DATA_DIR = "F:\\Dataset"
TFDS_SPLITS = ['train', 'validation', 'test']

# Path for generated sequnce sample 
SEQ_SAMPLE_PATH = "F:\\Dataset\\drum"

# Frame settings
FRAME_TIME = 10
SAMPLE_RATE = 16000
FRAME_SIZE = int(FRAME_TIME / 1000.0 * SAMPLE_RATE)

PITCH_LIST = [22, 26, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59]
PITCH_NUM = len(PITCH_LIST)
INDEX_DICT = dict(zip(PITCH_LIST, list(range(PITCH_NUM))))