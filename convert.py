import subprocess
import os

if __name__ == '__main__':
    directory = '/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/flt'
    files = os.listdir(directory)

    for file in files:
        if file.endswith('.csv'):
            file = directory + '/'  + file
            print(file)
            subprocess.run(['python', 'tools.py', '--model', 'flt', '--mode',
                        'convert', '--filepath', file])