# Settings for tensorflow datasets
TFDS_NAME = "groove/full-16000hz"
TFDS_DATA_DIR = "F:\\Dataset"
TFDS_SPLITS = ['train', 'validation', 'test']

# Path for preprocessed sequence sample 
SEQ_SAMPLE_PATH = "C:\\Users\\NKWBTB\\Desktop\\drum"

USE_SYNTH_AUDIO = True
USE_MIXUP = True
MIXUP_NUM = 5000
MIXUP_THRESH = 0.43

# Frame settings
MIN_SPLIT_LENGTH = 5
MAX_SPLIT_LENGTH = 12

FRAME_TIME = 10
SAMPLE_RATE = 16000
FRAME_SIZE = int(FRAME_TIME / 1000.0 * SAMPLE_RATE)

PITCH_LIST = [22, 26, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59]
PITCH_NUM = len(PITCH_LIST)
INDEX_DICT = dict(zip(PITCH_LIST, list(range(PITCH_NUM))))

DEFAULT_VELOCITY = 80
DEFAULT_QPM = 120

SPECTROGRAM = True
N_MELS = 250
FMIN = 30.0
HTK = True

NUM_EPOCHS = 60
BATCH_SIZE = 16
INPUT_SIZE = N_MELS if SPECTROGRAM else FRAME_SIZE

# Hack to allow python to pick up the newly-installed fluidsynth lib. 
# Tested under fluidsynth-1.1.9
# Some changes were made to pyfluidsynth (181-198 lines are commented)
import ctypes.util
orig_ctypes_util_find_library = ctypes.util.find_library
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth-1.dll'
  else:
    return orig_ctypes_util_find_library(lib)
ctypes.util.find_library = proxy_find_library
