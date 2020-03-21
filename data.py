import config as cfg
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import magenta.music as mm
import magenta
from bokeh.io import export_png

tf.enable_eager_execution()

def parse_sequence(sequence):
    notes = {'start_time': [], 'end_time': [], 'pitch': [], 'velocity': [], 'instrument': [], 'is_drum': []}
    for note in sequence.notes:
      notes['start_time'].append(note.start_time)
      notes['end_time'].append(note.end_time)
      notes['pitch'].append(note.pitch)
      notes['velocity'].append(note.velocity)
      notes['instrument'].append(note.instrument)
      notes['is_drum'].append(note.is_drum)
    
    total_time = sequence.total_time
    qpm = sequence.tempos[0].qpm

    return notes, total_time, qpm

def preprocess(save_path=cfg.SEQ_SAMPLE_PATH, frame_size=cfg.FRAME_SIZE):
    '''Preprocess the dataset into samples in sequence_level
    
    Args:
        save_path: path to save the sample files
        frame_size: number of point in a frame
        start_only: mark the label as 1 only at the frame with in the start_time
    
    Outputs:
        Pickle files of a dict, each file contains one sequence, for example:

        {'wave': [...], '{#Pitch_Number}': [...], '{#Pitch_Number}-onset': [...], ...}
    '''
    dataset = tfds.load(name=cfg.TFDS_NAME, data_dir=cfg.TFDS_DATA_DIR)

    for split in cfg.TFDS_SPLITS:
        dir = os.path.join(save_path, str(frame_size), split)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        for features in dataset[split]:
            midi = features['midi'].numpy()
            audio = features['audio'].numpy()
            sequence = mm.midi_to_note_sequence(midi)
            notes, total_time, qpm = parse_sequence(sequence)
            # fig = mm.plot_sequence(sequence, show_figure=False)
            # export_png(fig, filename="plot.png")
            assert(False)
            #sys.exit(0)

def main():
    preprocess()        

if __name__ == '__main__':
    main()

