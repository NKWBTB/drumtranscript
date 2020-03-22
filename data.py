import config as cfg
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import magenta.music as mm
import magenta
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
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

def preprocess(save_path=cfg.SEQ_SAMPLE_PATH, frame_size=cfg.FRAME_SIZE, frame_time = cfg.FRAME_TIME):
    '''Preprocess the dataset into samples in sequence_level
    
    Args:
        save_path: path to save the sample files
        frame_size: number of point in a frame
        frame_time: time period of a frame in millisecond(ms)
    
    Outputs:
        Pickle files of a dict, each file contains one sequence, for example:

        {'Frames': np.array((frame_num, frame_size)), 
        'Activation': np.array((frame_num, PITCH_NUM)), 
        'Onset': np.array((frame_num, PITCH_NUM))}
    '''
    dataset = tfds.load(name=cfg.TFDS_NAME, data_dir=cfg.TFDS_DATA_DIR)

    for split in cfg.TFDS_SPLITS:
        dir = os.path.join(save_path, str(frame_time) + '_ms', split)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        print('In ' + split)

        cnt = 0
        for features in dataset[split]:
            cnt += 1
            if cnt % 10 == 0:
                print(cnt)
            
            midi = features['midi'].numpy()
            audio = features['audio'].numpy()
            sequence = mm.midi_to_note_sequence(midi)
            
            # Padding audio to generate frames
            pad_num = frame_size - audio.shape[0] % frame_size
            if pad_num:
                audio = np.concatenate((audio, np.zeros((pad_num), dtype=audio.dtype)))
            frame_num = int(audio.shape[0]/frame_size)
            frames = np.reshape(audio, (frame_num, frame_size))
            
            # Sort the notes by start_time and end_time to do a sweep
            notes = [[note.start_time, note.end_time, cfg.INDEX_DICT[note.pitch]] for note in sequence.notes]
            # pitch = set([cfg.INDEX_DICT[note.pitch] for note in sequence.notes])
            # print(pitch)
            notes_dict = dict(zip(list(range(len(notes))), notes))
            start_dict = {}
            end_dict = {}

            for i in notes_dict:
                start_time = notes_dict[i][0]
                end_time = notes_dict[i][1]
                start_dict.setdefault(start_time, set()).add(i)
                end_dict.setdefault(end_time, set()).add(i)

            start_times = sorted(start_dict.keys())
            end_times = sorted(end_dict.keys())
            
            # Generate activation and onset matrix
            last_time = end_times[-1]
            t, l, r = 0, 0, 0
            status = set()
            activation = np.zeros((frame_num, cfg.PITCH_NUM), dtype=int)
            onset = np.zeros((frame_num, cfg.PITCH_NUM), dtype=int)
            for i in range(frame_num):
                st = i * frame_time
                ed = st + frame_time
                while l < len(start_times) and start_times[l] >= st and start_times[l] < ed:
                    note_list = start_dict[start_times[l]]
                    status.update(note_list)
                    l += 1
                    for j in note_list:
                        onset[i, notes_dict[j][2]] = 1
                
                for j in status:
                    activation[i, notes_dict[j][2]] = 1

                while r < len(end_times) and end_times[r] >= st and end_times[r] < ed:
                    status.discard(end_dict[end_times[r]])
                    r += 1
            
            '''
            if frame_num > 1000:
                x1mesh, x2mesh = np.meshgrid(np.arange(0, 1000),
                            np.arange(0, cfg.PITCH_NUM))
                print(cfg.PITCH_NUM, x1mesh.shape, onset.T.shape)
                plt.yticks(np.arange(0, cfg.PITCH_NUM))
                plt.pcolormesh(x1mesh, x2mesh, onset.T[:,:1000])
                plt.show()
                plt.close()
            '''
            # Save the sample to a file
            sample = {'Frames': frames, 'Activation': activation, 'Onset': onset}

            with open(os.path.join(dir, str(cnt) + '.pickle'), 'wb') as f:
                pickle.dump(sample, f)
            
        print('Total ' + str(cnt))
        
def main():
    preprocess()        

if __name__ == '__main__':
    main()

