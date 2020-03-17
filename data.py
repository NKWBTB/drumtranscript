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

def main():
    dataset = tfds.load(name="groove/full-16000hz", data_dir="F:\\Dataset")
    for features in dataset['train'].take(1):
        print(features.keys(), features['id'])
        midi = features['midi'].numpy()
        print(len(midi))

        sequence = mm.midi_to_note_sequence(midi)
        # fig = mm.plot_sequence(sequence, show_figure=False)
        # export_png(fig, filename="plot.png")

        mm.sequence_proto_to_midi_file(sequence, 'notes.mid')

        notes, total_time, qpm = parse_sequence(sequence)
        

if __name__ == '__main__':
    main()

