
import config as cfg
from data import *
from utils import *
import magenta.music as mm
import os

def main():
    test_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'test')
    test_files = list_files(test_path, 'pickle')

    sequence1 = mm.midi_file_to_note_sequence(os.path.join(test_path, '1.mid'))
    print(sequence1.tempos[0])

    fig = mm.plot_sequence(sequence1, show_figure=False)
    export_png(fig, filename="test.png")

    print(test_files[0])
    sample = read_sample(test_files[0])
    print(sample['Activation'].shape)
    # plot_matrix(sample['Activation'].T)
    # plot_matrix(sample['Onset'].T)
    sequence2 = matrix2sequence(sample['Activation'])
    mm.sequence_proto_to_midi_file(sequence2, 'test.mid')
    fig2 = mm.plot_sequence(sequence2, show_figure=False)
    export_png(fig2, filename="test_gen.png")

if __name__ == "__main__":
    main()