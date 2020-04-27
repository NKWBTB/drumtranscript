import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Dropout, Reshape
from tensorflow.keras.models import Model
import config as cfg
import os
from data import read_sample
from utils import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class BaseModel:
    def __init__(self):
        pass

    def train(self):
        ''' Train the model '''
        raise NotImplementedError

    def predict(self, input):
        ''' Predict the activation and onset using the model

        Returns:
            activation, onset(optional, default=None)
        '''
        raise NotImplementedError

    def evaluate(self):
        ''' Report the precision/recall/f-measure of the model'''
        raise NotImplementedError

    def save(self, path):
        ''' Save the model to the path'''
        raise NotImplementedError

    def load(self, path):
        ''' Load the model from the path'''
        raise NotImplementedError


class BiLSTM(BaseModel):
    def __init__(self):
        self.checkpoint_path = "models/Bi-LSTM/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None

    def build_model(self, input_shape=(None, cfg.INPUT_SIZE)):
        input = Input(shape=input_shape, dtype='float32')
        x = Dense(128, activation='relu')(input)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

    def data_generator(self, filelist, batch_size=1):
        while 1:
            random.shuffle(filelist)
            num = len(filelist)
            cnt = 0
            x, y = [], []
            for i in range(num):
                sample = read_sample(filelist[i])
                x.append(sample['Frames'])
                y.append(sample['Onset'])
                cnt += 1
                if cnt == batch_size or i == num-1:
                    x = np.array(x)
                    y = np.array(y)
                    cnt = 0
                    yield (x, y)
                    x, y = [], []
    
    def last_epoch(self):
        checkpoints = os.listdir(self.checkpoint_dir)
        epochs = [int(epoch.split('.')[0]) for epoch in checkpoints]
        epochs.append(0)
        last = max(epochs)
        return last

    def train(self, train_files, val_files, batch_size=cfg.BATCH_SIZE):
        num_epochs = cfg.NUM_EPOCHS
        trainning_steps = int(math.ceil(len(train_files) / batch_size))
        # checkpoint settings
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, 
            verbose=1)
        '''
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=trainning_steps,
            decay_rate=0.98,
            staircase=True)
        '''
        model = None
        init_epoch = 0
        if not os.path.exists(self.checkpoint_dir) or self.last_epoch() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.build_model()    
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            self.model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            init_epoch = self.last_epoch()
            print('\nLoad existing model from epoch %i\n' % init_epoch)
            self.load(os.path.join(self.checkpoint_dir, str(init_epoch)+'.h5'))

        self.model.summary()
        history = self.model.fit_generator(self.data_generator(train_files, batch_size), 
                                        epochs=num_epochs, initial_epoch=init_epoch, steps_per_epoch=trainning_steps, 
                                        validation_data=self.data_generator(val_files), validation_steps=len(val_files),
                                        callbacks=[cp_callback, keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                            min_delta=0,
                                                                                            patience=10,
                                                                                            verbose=0,
                                                                                            mode='auto')])
        plt.figure(figsize=(5, 2))
        try:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig('history.png')
        except Exception as e:
            print("\nException caught:\n", e, "\n")
        finally:
            plt.close()

        return history

    def predict(self, frames, threshold=None):
        if frames.ndim == 2:
            frames = np.array([frames], dtype=np.float32)
        if threshold is None:
            return self.model.predict(frames)
        activation = np.array(self.model.predict(frames) >= threshold, dtype=int)
        return activation, None
    
    def evaluate(self, val_files, test_files, thresholds=np.arange(0, 1, 0.1)):
        import mir_eval
        from data import matrix2sequence, parse_sequence, measure
        
        def test(files, thresholds):
            best_file, best_f = None, 0
            metrics = np.zeros((thresholds.shape[0], cfg.PITCH_NUM, 3), dtype=float)
            cnt = 0
            for file in files:
                cnt += 1
                if cnt % 10 == 0:
                    print(cnt)
                sample = read_sample(file)
                gt_notes = sample['Sequence']

                y = self.predict(sample['Frames'])
                for i in range(thresholds.shape[0]):
                    y_pred = np.array(y >= thresholds[i], dtype=int)
                    pred_sequence = matrix2sequence(y_pred[0], onset=y_pred[0])
                    pred_notes = parse_sequence(pred_sequence)

                    score, _, _, average_f = measure(gt_notes, pred_notes)
                    metrics[i] += score

                    if average_f > best_f and len(sample['Audio']) / float(cfg.SAMPLE_RATE) > 3:
                        best_f = average_f
                        best_file = file
            
            return metrics, best_file, best_f

        metrics, _, _ = test(val_files, thresholds)
        p, r, f = [], [], []
        for i in range(thresholds.shape[0]):
            TP = float(np.sum(metrics[i, :, 0]))
            FP = np.sum(metrics[i, :, 1])
            FN = np.sum(metrics[i, :, 2])
            p.append(TP / (TP + FP))
            r.append(TP / (TP + FN))
            f.append(2*p[i]*r[i]/(p[i]+r[i]))
    
        plot_line(f, thresholds, 'thresh.png', 'Threshold', 'F-measure')
        plot_line(r, p, 'pr.png', 'Precision', 'Recall')
        save_array(self.__class__.__name__ + '_prf.csv', [p, r, f])

        best_idx = np.argmax(f)
        print('Best_threshold', thresholds[best_idx])
        best_thresh = thresholds[best_idx]

        metrics, best_file, best_f = test(test_files, thresholds=np.array([best_thresh]))     
        p = metrics[0, :, 0] / (metrics[0, :, 0] + metrics[0, :, 1])
        r = metrics[0, :, 0] / (metrics[0, :, 0] + metrics[0, :, 2])
        f = 2*p*r/(p+r)
        plot_bar(p, np.arange(cfg.PITCH_NUM), cfg.PITCH_LIST, 'precicion.png', 'Pitch', 'Precicion')
        plot_bar(r, np.arange(cfg.PITCH_NUM), cfg.PITCH_LIST, 'recall.png', 'Pitch', 'Recall')
        plot_bar(f, np.arange(cfg.PITCH_NUM), cfg.PITCH_LIST, 'fscore.png', 'Pitch', 'F-measure')
        print('Best test: ', best_file, best_f)
        sample = read_sample(best_file)
        save_wav('best.wav', sample['Audio'], cfg.SAMPLE_RATE)
        return metrics

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

class SimpleLSTM(BiLSTM):
    def __init__(self):
        self.checkpoint_path = "models/LSTM/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None
    
    def build_model(self, input_shape=(None, cfg.INPUT_SIZE)):
        input = Input(shape=input_shape, dtype='float32')
        x = Dense(128, activation='relu')(input)
        x = LSTM(128, return_sequences=True)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

class SimpleDNN(BiLSTM):
    def __init__(self):
        self.checkpoint_path = "models/DNN/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None
    
    def build_model(self, input_shape=(None, cfg.INPUT_SIZE)):
        input = Input(shape=input_shape, dtype='float32')
        x = Dense(128, activation='relu')(input)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

class OaF_Drum(BiLSTM):
    def __init__(self):
        self.checkpoint_path = "models/OaF_Drum/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None
    
    def build_model(self, input_shape=(None, cfg.INPUT_SIZE), use_lstm=False, use_dropout=True):
        input = Input(shape=input_shape, dtype='float32')
        reshape = Reshape((-1, cfg.INPUT_SIZE, 1))(input)
        x = Conv2D(16, (3,3), padding='SAME', activation='relu')(reshape)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3,3), padding='SAME', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 2), strides=(1, 2))(x)
        if use_dropout:
            x = Dropout(0.25)(x)
        x = Conv2D(32, (3,3), padding='SAME', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 2), strides=(1, 2))(x)
        if use_dropout:
            x = Dropout(0.25)(x)
        dim = x.get_shape()
        x = Reshape((-1, int(dim[2]*dim[3])))(x)
        x = Dense(256, activation='relu')(x)
        if use_dropout:
            x = Dropout(0.5)(x)
        if use_lstm:
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

if __name__ == "__main__":
    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = list_files(val_path, 'pickle')
    test_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'test')
    test_files = list_files(test_path, 'pickle')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    model = OaF_Drum()
    #model.train(train_files, val_files, batch_size=16)
    model.load('models/OaF_Drum_epoch31.h5')
    model.evaluate(val_files, test_files, thresholds=np.arange(0, 1, 0.1))
    #model = SimpleDNN()
    #model.train(train_files, val_files)