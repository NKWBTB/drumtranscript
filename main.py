
import config as cfg
from data import *
from utils import *
import magenta.music as mm
import os
import numpy as np
from model import *

def main():
    test_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'test')
    test_files = list_files(test_path, 'pickle')
    idx = 40
    print(test_files[idx])
    sample = read_sample(test_files[idx])
    
    model = BiLSTM()
    model.load(os.path.join(model.checkpoint_dir, '17.h5'))    
    '''
    activation, onset = model.predict(sample['Frames'])

    sequence1 = mm.midi_file_to_note_sequence(test_files[idx].split('.')[0] + '.mid')
    print(sequence1.tempos[0])
    fig = mm.plot_sequence(sequence1, show_figure=False)
    export_png(fig, filename="test.png")

    sequence2 = matrix2sequence(sample['Activation'])
    mm.sequence_proto_to_midi_file(sequence2, 'test.mid')
    fig2 = mm.plot_sequence(sequence2, show_figure=False)
    export_png(fig2, filename="test_gen.png")

    sequence3 = matrix2sequence(activation[0])
    mm.sequence_proto_to_midi_file(sequence3, 'pred.mid')
    fig3 = mm.plot_sequence(sequence3, show_figure=False)
    export_png(fig3, filename="pred.png")
    '''

    thresholds, precision, recall, f_measure = model.evaluate(test_files)
    print(thresholds, precision, recall, f_measure)
    precision = np.mean(precision, axis=1)
    recall = np.mean(recall, axis=1)
    f_measure = np.mean(f_measure, axis=1)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(thresholds, f_measure)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()