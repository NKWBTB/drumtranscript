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