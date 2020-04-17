import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Dropout, Reshape
from tensorflow.keras.models import Model
import config as cfg
import os
from data import read_sample
from utils import list_files, save_wav
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
        x = Bidirectional(LSTM(128, return_sequences=True))(input)
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
        
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.98,
            staircase=True)

        model = None
        init_epoch = 0
        if not os.path.exists(self.checkpoint_dir) or self.last_epoch() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.build_model()    
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
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
                                                                                            patience=3,
                                                                                            verbose=0,
                                                                                            mode='auto')])
        return history

    def predict(self, frames, threshold=0.5):
        if frames.ndim == 2:
            frames = np.array([frames], dtype=np.float32)
        activation = np.array(self.model.predict(frames) >= threshold, dtype=int)
        return activation, None
    
    def evaluate(self, test_files, thresholds=np.array([0.5])):
        import mir_eval
        from data import matrix2sequence, parse_sequence
        metrics = np.zeros((thresholds.shape[0], cfg.PITCH_NUM, 3), dtype=float)
        best_file, best_f = None, 0
        for i in range(thresholds.shape[0]):
            cnt = 0
            sample_num = np.zeros((cfg.PITCH_NUM, 1))
            for file in test_files:
                cnt += 1
                if cnt % 10 == 0:
                    print(cnt)
                sample = read_sample(file)
                y_pred, _ = self.predict(sample['Frames'], threshold=thresholds[i])
                pred_sequence = matrix2sequence(y_pred[0], onset=y_pred[0])

                gt_notes = sample['Sequence']
                pred_notes = parse_sequence(pred_sequence)

                sum, num = 0, 0
                for j in cfg.PITCH_LIST:
                    if (not j in gt_notes) and (not j in pred_notes):
                        continue
                    ref_intervals = gt_notes.get(j, np.zeros((0, 2), dtype=float))
                    est_intervals = pred_notes.get(j, np.zeros((0, 2), dtype=float))
                    p, r, f = mir_eval.transcription.onset_precision_recall_f1(ref_intervals, est_intervals)

                    k = cfg.INDEX_DICT[j]
                    sample_num[k, 0] += 1
                    metrics[i, k, 0] += p
                    metrics[i, k, 1] += r
                    metrics[i, k, 2] += f
                    sum += f
                    num += 1
                if sum / float(num) > best_f and len(sample['Audio']) / float(cfg.SAMPLE_RATE) > 3:
                    best_f = sum / float(num)
                    best_file = file
                
            metrics[i] /= sample_num
        
        plt.figure(figsize=(10, 5))
        plt.ylim((0.0, 1.0))
        plt.xticks(np.arange(cfg.PITCH_NUM), labels=cfg.PITCH_LIST)
        plt.bar(np.arange(cfg.PITCH_NUM), metrics[0, :, 2])
        #plt.show()
        plt.savefig('eval.png')
        plt.close()
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
        x = LSTM(128, return_sequences=True)(input)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

class OaF_Drum(BiLSTM):
    def __init__(self):
        self.checkpoint_path = "models/OaF_Drum/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None
    
    def build_model(self, input_shape=(None, cfg.INPUT_SIZE), use_lstm=True):
        input = Input(shape=input_shape, dtype='float32')
        reshape = Reshape((-1, cfg.INPUT_SIZE, 1))(input)
        x = Conv2D(16, (3,3), padding='SAME', activation='relu')(reshape)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3,3), padding='SAME', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 2), strides=(1, 2))(x)
        x = Dropout(0.75)(x)
        x = Conv2D(32, (3,3), padding='SAME', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 2), strides=(1, 2))(x)
        x = Dropout(0.75)(x)
        dim = x.get_shape()
        x = Reshape((-1, int(dim[2]*dim[3])))(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        if use_lstm:
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            x = Dropout(0.5)(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

if __name__ == "__main__":
    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = list_files(val_path, 'pickle')
    
    model = OaF_Drum()
    model.train(train_files, val_files, batch_size=16)
