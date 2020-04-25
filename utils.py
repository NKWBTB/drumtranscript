import os
import matplotlib.pyplot as plt
import numpy as np
import inspect
import librosa

def plot_array(array, x_range = None, subplot = None):
    if type(x_range) == type(None):
        x_range = np.arange(array.shape[0])
    if type(subplot) == type(None):
        plt.figure()
    else:
        plt.subplot(subplot)
    plt.plot(x_range, array)
    if type(subplot) == type(None):
        plt.show()
        plt.close()

def plot_matrix(mat, x_range = None, y_range = None):
    if type(x_range) == type(None):
        x_range = np.arange(mat.shape[1])
    if type(y_range) == type(None):
        y_range = np.arange(mat.shape[0])
    x1mesh, x2mesh = np.meshgrid(x_range, y_range)
    plt.figure()
    plt.yticks(y_range)
    plt.pcolormesh(x1mesh, x2mesh, mat[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1])
    plt.show()
    plt.close()

def plot_bar(y, xrange, xlabel, filename, xtitle = None, ytitle = None):
    plt.figure(figsize=(10, 5))
    plt.ylim((0.0, 1.0))
    plt.xticks(xrange, labels=xlabel)
    plt.bar(xrange, y)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    # plt.show()
    plt.savefig(filename)
    plt.close()

def plot_line(y, x, filename, xtitle = None, ytitle = None):
    plt.figure(figsize=(5, 5))
    plt.ylim((0.0, 1.0))
    plt.plot(x, y)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    # plt.show()
    plt.savefig(filename)
    plt.close()

def list_files(path, extension = ''):
    files = os.listdir(path)
    filtered = []
    for file in files:
        if file.split('.')[-1] == extension:
            filtered.append(os.path.join(path, file))
    return filtered

def list_class(module):
    return [m[0] for m in inspect.getmembers(module, inspect.isclass) \
        if m[1].__module__ == module.__name__]

def load_wav(path, sr):
    wav, _ = librosa.load(path, sr)
    wav = librosa.util.normalize(wav)
    return wav

def save_wav(path, audio, sr):
    librosa.output.write_wav(path, audio, sr)