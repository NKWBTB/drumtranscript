import tensorflow as tf
import tensorflow_datasets as tfds
import magenta.music as mm
import magenta
from bokeh.io import export_png

tf.enable_eager_execution()

def main():
    dataset = tfds.load(name="groove/full-16000hz", data_dir="F:\\Dataset")
    for features in dataset['train'].take(1):
        print(features.keys())
        midi = features['midi'].numpy()

        notes = mm.midi_to_note_sequence(midi)
        fig = mm.plot_sequence(notes, show_figure=False)
        export_png(fig, filename="plot.png")

        mm.sequence_proto_to_midi_file(notes, 'notes.mid')

if __name__ == '__main__':
    main()

