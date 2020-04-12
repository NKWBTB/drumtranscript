import config as cfg
import data
from utils import *
import magenta.music as mm
import os
import numpy as np
import model
import argparse
import librosa
import scipy

def test():
    test_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'test')
    test_files = list_files(test_path, 'pickle')
    idx = 40
    print(test_files[idx])
    sample = data.read_sample(test_files[idx])
    
    m = model.OaF_Drum()
    m.load(os.path.join(m.checkpoint_dir, '8.h5'))    
    
    onset, _ = m.predict(sample['Frames'])

    sequence1 = mm.midi_file_to_note_sequence(test_files[idx].split('.')[0] + '.mid')
    print(sequence1.tempos[0])
    fig = mm.plot_sequence(sequence1, show_figure=False)
    data.export_png(fig, filename="test.png")

    sequence2 = data.matrix2sequence(sample['Onset'], onset=sample['Onset'])
    mm.sequence_proto_to_midi_file(sequence2, 'test.mid')
    fig2 = mm.plot_sequence(sequence2, show_figure=False)
    data.export_png(fig2, filename="test_gen.png")

    sequence3 = data.matrix2sequence(onset[0], onset=onset[0])
    mm.sequence_proto_to_midi_file(sequence3, 'pred.mid')
    fig3 = mm.plot_sequence(sequence3, show_figure=False)
    data.export_png(fig3, filename="pred.png")