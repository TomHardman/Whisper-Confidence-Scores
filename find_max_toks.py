import pickle
from collections import defaultdict
from tqdm import tqdm

path = '/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/test_gec_beam5_layer-1_data.pkl'

if __name__ == '__main__':
    max_toks = -1
    tok_counts = defaultdict(int)
    corr_counts = defaultdict(int)
    print('opening file')

    with open(path, 'rb') as f:
        word_dicts = pickle.load(f)
    
    print('opened file')
    
    for wd in tqdm(word_dicts):
        tok_counts[len(wd['labels'])] += 1
        if wd['labels'][0] == 0:
            corr_counts[len(wd['labels'])] += 1
        if len(wd['labels']) > 3:
            print(wd['word'], wd['labels'])

    print(tok_counts, corr_counts)