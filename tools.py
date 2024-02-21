from sklearn.metrics import precision_recall_curve as pr_curve, auc 
from sklearn.calibration import calibration_curve as cal_curve
from cem_predict import predict_confidence_scores
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def csv_to_pickle(path):
    """
    Reads CSV file and stores as pickled pandas dataframe
    """
    name = path.split('/')[-1].split('.')[0]
    chunks = []

    # Read the CSV file into a Pandas DataFrame
    for chunk in pd.read_csv(path, chunksize=10000):
        chunks.append(chunk)

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    # Save as pickle file
    pickle_file = f'exp/cem_data/{name}.pkl'
            
    with open(pickle_file, 'wb') as f:
        pickle.dump(df, f)


def invert_labels_and_probs(path):
    '''
    We want to detect the event which a token is incorrect rather than correct,
    so we must invert the probablities and labels in the table to identify
    this.
    '''
    with open(path, 'rb') as f:
            df = pickle.load(f)
    
    df.iloc[:, -2] = 1 - df.iloc[:, -2] # invert probablities
    df.iloc[:, -1] = abs(df.iloc[:, -1] - 1) # convert 1 to zero
    out_path = path.split('.')[0] + '_inverted.pkl'
    
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)


def get_pr(path):
    with open(path, 'rb') as f:
            data = pickle.load(f)

    # Extract the second-to-last and last columns into Python lists
    try:
        confs = data.iloc[:, -2].tolist()  # Second-to-last column
        y = data.iloc[:, -1].tolist()  # Last column

    except AttributeError:
        confs = data[0]
        y = data[1]
    p, r, t = pr_curve(y, confs)
    return p, r


def f_score(p, r, beta=0.5):
    num = 1+beta**2
    denom = 1/p + beta**2/r
    return num/denom


def plot(paths, labels, model):
    fig, ax = plt.subplots()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    for path, label in zip(paths, labels):
        p, r = get_pr(path)
        f = f_score(np.array(p), np.array(r))
        best_f = np.max(f)
        ax.plot(r, p, label=f'{label} AUC: {auc(r, p):.5f}, Best F: {best_f:.5f}')
        plot_reliability_diagram(path, label, model)
    ax.legend(loc='lower left')         
    fig.savefig(f'exp/pr_curves/pr_{model}.png')


def rel_diagram(confs, y, n=20):
    count = [0]*n
    ECE = 0
    UCE = 0
    bar_count = 0
    nums = [0]*n
    accs = np.linspace(1.0/n/2, 1.0-1.0/n/2, n)
    for i in range(n):
        left = 1.0/n*i
        right = 1.0/n*(i+1)
        mid = accs[i]
        
        correct = sum(map(lambda p,l: p>left and p <=right and l >= 0.5, confs, y))
        incor = sum(map(lambda p,l: p>left and p <=right and l <= 0.5, confs, y))
        
        if correct+incor !=0:
            accuracy = correct/(correct+incor)
            ECE += abs(accuracy - mid)*(correct+incor)
            UCE += abs(accuracy - mid)
            bar_count += 1

        else:
            accuracy = 0
        
        count[i] = accuracy
        nums[i]=correct+incor
    ECE = ECE/sum(nums)
    UCE = UCE/bar_count

    return ECE, UCE, accs, count


def plot_reliability_diagram(path, label, model, n=20, plot=True):
    title = label + ' ' + model
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Extract the second-to-last and last columns into Python lists
    try:
        confs = data.iloc[:, -2].tolist()  # Second-to-last column
        y = data.iloc[:, -1].tolist()  # Last column
    except AttributeError:
        confs = data[0]
        y = data[1]
    
    ECE, UCE, accs, count = rel_diagram(confs, y, n)

    if plot:
        fig, ax = plt.subplots()
        ax.bar(accs, count, label=f'ECE: {ECE:.5f}, UCE: {UCE:.5f}', width=accs[1]-accs[0], alpha=0.8)
        ax.plot(accs, accs, '--', color='orange', label='perfect calibration') # calibration line
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('True Probability')
        ax.set_title(label)
        ax.legend()
        fig.savefig(f'exp/rel_diagrams/{title.replace(" ", "_")}')
    
    return ECE, UCE


def hist_plot(path):
    with open(path, 'rb') as f:
            data = pickle.load(f)

    confs = data.iloc[:, -2].tolist()  # Second-to-last column
    y = data.iloc[:, -1].tolist()  # Last column

    # Separate confidences based on target values
    confidences_target0 = [confs[i] for i in range(len(confs)) if y[i]==0]
    confidences_target1 = [confs[i] for i in range(len(confs)) if y[i]==1]

    # Plot histograms with different colors for each target
    plt.hist([confidences_target0, confidences_target1], bins=10, color=['blue', 'red'], label=['0', '1'])
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    plt.savefig('exp/hist.png')

def list_type(arg):
    return [x for x in arg.split(',')]

def label_list_type(arg):
    return [' '.join(x.split('_')) for x in arg.split(',')]
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tools for analysis')
    parser.add_argument('--model', type=str, default='flt', help='Model to use: gec, flt or dsf')
    parser.add_argument('--mode', type=str, default='plot')
    parser.add_argument('--filepath', type=str, default='exp/cem_data/gec/test_gec_beam5_inverted.pkl')
    parser.add_argument('--pred_paths', type=list_type, help='Paths to prediction files',
                        default= '/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_models/gec/train_gec_beam5_SA20_50_2_50_2_inverted/16_pw1.0_usedec0/epoch22_f0.45060_loss0.255_auc0.41091_UCE0.0315_preds.pkl,/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_models/gec/train_gec_beam5_SA10_30_2_50_2_inverted/16_pw1.0_usedec0/epoch14_f0.44765_loss0.252_auc0.41714_UCE0.0335_preds.pkl')
    parser.add_argument('--labels', type=label_list_type, default='CEM_Aug+,CEM_Aug')
    parser.add_argument('--only_baseline', default=0)
    args = parser.parse_args()

    if args.mode == 'convert':
        csv_path = args.filepath
        csv_to_pickle(csv_path)
    
    elif args.mode == 'invert':
        pickle_path = args.filepath
        invert_labels_and_probs(pickle_path)

    else:
        model_names = {'gec': 'Whisper GEC',
                       'flt': 'Whisper Fluent',
                       'dsf': 'Whisper Disfluent'}
        
        if not args.only_baseline:
            paths = [args.filepath] + args.pred_paths
            print(paths)
            labels = ['Softmax Baseline'] + args.labels
        else:
            paths = [args.filepath]
            labels = ['Softmax Baseline']
        
        hist_plot(args.filepath)
        model = model_names.get(args.model)
        plot(paths, labels, model)
        
