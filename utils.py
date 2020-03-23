import os

def list_files(path, extension = ''):
    files = os.listdir(path)
    filtered = []
    for file in files:
        if file.split('.')[-1] == extension:
            filtered.append(file)
    return filtered