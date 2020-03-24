import config as cfg
import utils
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import magenta.music as mm
from magenta.music import midi_synth
from magenta.music.protobuf import music_pb2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bokeh.io import export_png
import pickle
import scipy

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

def read_sample(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
        return sample

def matrix2sequence(activation, frame_time=cfg.FRAME_TIME, onset=None):
    '''Given acvtivation and onset(optional) matrix to recover the notesequence

    Args:
        activation: activation matrix
        frame_time: time period of a frame in millisecond(ms)
        onset: onset matrix, optional

    Returns:
        A notesequence recovered from the inputs.
    '''
    activation = np.append(activation, [np.zeros(activation[0].shape)], axis=0)
    if type(onset) != type(None):
        onset = np.append(onset, [np.zeros(onset[0].shape)], axis=0)
        activation = np.logical_or(activation, onset)
    
    sequence = music_pb2.NoteSequence()
    frame_num = activation.shape[0]
    start_times = {}
    
    def handle_inactive(t, i, j):
        pitch = cfg.PITCH_LIST[j]
        if pitch in start_times:
            st = start_times[pitch]
            sequence.notes.add(pitch=pitch, 
                            start_time=st,
                            end_time=t,
                            is_drum=True,
                            instrument=10,
                            velocity=cfg.DEFAULT_VELOCITY)
            del start_times[pitch]

    def handle_active(t, i, j):
        pitch = cfg.PITCH_LIST[j]
        if type(onset) != type(None):
            if onset[i][j]:
                if pitch in start_times:
                    handle_inactive(t, i, j)
                start_times[pitch] = t
        else:
            if not pitch in start_times:
                start_times[pitch] = t

    for i in range(frame_num):
        t = frame_time * (i + 0.5) / 1000.0
        for j in range(cfg.PITCH_NUM):
            if activation[i, j]:
                handle_active(t, i, j)
            else:
                handle_inactive(t, i, j)
    
    sequence.total_time = activation.shape[0] * frame_time / 1000.0
    return sequence

def audio2frame(audio, frame_size):
    # Padding audio
    pad_num = frame_size - audio.shape[0] % frame_size
    if pad_num:
        audio = np.concatenate((audio, np.zeros((pad_num), dtype=audio.dtype)))
    frame_num = int(audio.shape[0]/frame_size)
    frames = np.reshape(audio, (frame_num, frame_size))
    return frames, frame_num

def preprocess(save_path=cfg.SEQ_SAMPLE_PATH, frame_size=cfg.FRAME_SIZE, frame_time=cfg.FRAME_TIME):
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

    synth_except = False

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
            frames, frame_num = audio2frame(audio, frame_size)
            sequence = mm.midi_to_note_sequence(midi)
            # fig = mm.plot_sequence(sequence, show_figure=False)
            # export_png(fig, filename=os.path.join(dir, "%i.png" % cnt))
            if not synth_except:
                try:
                    synthed_audio = midi_synth.fluidsynth(sequence, cfg.SAMPLE_RATE)
                    scipy.io.wavfile.write(os.path.join(dir,'%i_syn.wav' % cnt), cfg.SAMPLE_RATE, synthed_audio)
                except Exception as e:
                    print("\nException caught:\n", e, "\n")
                    synth_except = True
            scipy.io.wavfile.write(os.path.join(dir, '%i.wav' % cnt), cfg.SAMPLE_RATE, audio)
            mm.sequence_proto_to_midi_file(sequence, os.path.join(dir, '%i.mid' % cnt))

            '''
            plt.figure()
            utils.plot_array(audio, subplot=211)
            utils.plot_array(synthed_audio, subplot=212)
            plt.show()
            plt.close()
            '''
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
                st = i * frame_time / 1000.0
                ed = (i+1) * frame_time / 1000.0
                while l < len(start_times) and start_times[l] >= st and start_times[l] < ed:
                    note_list = start_dict[start_times[l]]
                    status.update(note_list)
                    l += 1
                    for j in note_list:
                        onset[i, notes_dict[j][2]] = 1
              
                for j in status:
                    activation[i, notes_dict[j][2]] = 1

                while r < len(end_times) and end_times[r] >= st and end_times[r] < ed:
                    note_list = end_dict[end_times[r]]
                    status = status - note_list
                    r += 1
            '''  
            if frame_num > 200:
                utils.plot_matrix(activation.T)
                assert(False)
            '''
            # Save the sample to a file
            sample = {'Frames': frames, 'Activation': activation, 'Onset': onset}

            with open(os.path.join(dir, str(cnt) + '.pickle'), 'wb') as f:
                pickle.dump(sample, f)
            
        print('Total ' + str(cnt))

def main():
    tf.enable_eager_execution()
    preprocess()        

if __name__ == '__main__':
    main()

