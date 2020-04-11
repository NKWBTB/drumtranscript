
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
    
    model = OaF_Drum()
    model.load(os.path.join(model.checkpoint_dir, '8.h5'))    
    
    onset, _ = model.predict(sample['Frames'])

    sequence1 = mm.midi_file_to_note_sequence(test_files[idx].split('.')[0] + '.mid')
    print(sequence1.tempos[0])
    fig = mm.plot_sequence(sequence1, show_figure=False)
    export_png(fig, filename="test.png")

    sequence2 = matrix2sequence(sample['Onset'], onset=sample['Onset'])
    mm.sequence_proto_to_midi_file(sequence2, 'test.mid')
    fig2 = mm.plot_sequence(sequence2, show_figure=False)
    export_png(fig2, filename="test_gen.png")

    sequence3 = matrix2sequence(onset[0], onset=onset[0])
    mm.sequence_proto_to_midi_file(sequence3, 'pred.mid')
    fig3 = mm.plot_sequence(sequence3, show_figure=False)
    export_png(fig3, filename="pred.png")
    
    # thresholds = np.arange(0, 1, 0.1)
    metrics = model.evaluate(test_files)
    #precision = np.mean(metrics[:,:,0], axis=1)
    #recall = np.mean(metrics[:,:,1], axis=1)
    #f_measure = np.mean(metrics[:,:,2], axis=1)

    #print(f_measure.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    #plt.plot(thresholds, f_measure)
    plt.xticks(cfg.PITCH_LIST)
    plt.bar(cfg.PITCH_LIST, metrics[0, :, 2])
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    main()