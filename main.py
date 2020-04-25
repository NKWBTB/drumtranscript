
import os
import argparse
import librosa
import scipy
import config as cfg
import utils

def transcribe(m, input, output, threshold):
    import magenta.music as mm
    import data
    wav = utils.load_wav(input, cfg.SAMPLE_RATE)
    frames, _ = data.audio2frame(wav, cfg.FRAME_SIZE, cfg.SPECTROGRAM)
    onset, _ = m.predict(frames, threshold)
    sequence = data.matrix2sequence(onset[0], onset=onset[0])
    mm.sequence_proto_to_midi_file(sequence, output)

def main():
    import tensorflow as tf
    tf.enable_eager_execution()
    import model
    import data

    models_available = utils.list_class(model)
    models_available.remove('BaseModel')

    parser = argparse.ArgumentParser(
            description="Transcribe the drum solo from a wave file to a midi file.")
    parser.add_argument('--task', '-t', required=True,
                        choices=['pre', 'train', 'test', 'trans'],
                        help="The task you want to execute. \
                        pre: preprocess the dataset;\
                        train: train the model;\
                        test: test the model performance on testset;\
                        trans: transcibe using a trained model.")
    parser.add_argument('--model', '-m',
                        choices=models_available,
                        help="The type of model used.")
    parser.add_argument('--model_path', '-p',
                        help="The model file used for 'trans' and 'test'.")
    parser.add_argument('--input', '-i',
                        help="The wave input file for transcribe, only useful when task being 'trans'.")
    parser.add_argument('--output', '-o', 
                        help="The midi output file for transcribe, only useful when task being 'trans'.")
    parser.add_argument('--threshold', '-T', default=0.5, type=float,
                        help="The threshold for transcibe, only useful when task being 'trans'.")
    args = parser.parse_args()

    if args.task == 'pre':
        data.preprocess()
        return

    tf.disable_eager_execution()
    if args.model is None:
        print('\n\nError: --model must be specified!!')
        return
    m = getattr(model, args.model)()

    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = utils.list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = utils.list_files(val_path, 'pickle')
    test_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'test')
    test_files = utils.list_files(test_path, 'pickle')
    
    if args.task == 'train':
        m.train(train_files, val_files)
        return
    
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            print('Model file does not exists.')
            return -1
        m.load(args.model_path)
    else:
        print('\n\nError: --model_path must be specified!!')
        return -1
    
    if args.task == 'test':
        m.evaluate(val_files, test_files)
    else:
        if args.input is None or args.output is None:
            print('\n\nError: --input and --output must be specified!!')
            return -1
        if not os.path.exists(args.input):
            print('\n\nError: Input wav file does not exists.')
            return -1
        transcribe(m, args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()