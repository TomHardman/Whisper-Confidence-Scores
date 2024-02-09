from sklearn.metrics import precision_recall_curve as pr_curve, auc 
from sklearn.calibration import calibration_curve as cal_curve
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch

def csv_to_pickle(path):
    """
    Reads CSV file and stores as pickled pandas dataframe
    """
    name = path.split('/')[-1].split('.')[0]

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(path)

    # Save as pickle file
    name = path.split('/')[-1].split('.')[0]
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
        reliability_diagram(path, label)
    ax.legend(loc='lower left')         
    fig.savefig(f'exp/pr_curves/pr_{model}.png')


def reliability_diagram(path, title, n=20):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Extract the second-to-last and last columns into Python lists
    try:
        confs = data.iloc[:, -2].tolist()  # Second-to-last column
        y = data.iloc[:, -1].tolist()  # Last column
    except AttributeError:
        confs = data[0]
        y = data[1]
    
    count = [0]*n
    ECE = 0
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
        else:
            accuracy = 0
        
        count[i] = accuracy
        nums[i]=correct+incor
    ECE = ECE/sum(nums)
    
    fig, ax = plt.subplots()
    ax.bar(accs, count, label=f'ECE: {ECE:.5f}', width=accs[1]-accs[0], alpha=0.8)
    ax.plot(accs, accs, '--', color='orange', label='perfect calibration') # calibration line
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('True Probability')
    ax.set_title(title)
    ax.legend()
    fig.savefig(f'exp/rel_diagrams/{title.replace(" ", "_")}')
          

if __name__ == '__main__':
    stage = 'invert'

    if stage == 'convert':
        csv_path = 'exp/cem_data/train_beam5_SA5_22_2_0_0.csv'
        csv_to_pickle(csv_path)
    
    elif stage == 'invert':
        pickle_path = 'exp/cem_data/dev_beam5.pkl'
        invert_labels_and_probs(pickle_path)

    else:
        pickle_path_base = 'exp/cem_data/dev_beam5.pkl'
        paths = ['exp/cem_data/test_beam5.pkl', 'exp/cem_data/dev_beam5.pkl', 
                 'exp/cem_data/train_beam5_SA5_22_2_0_0.pkl']
        labels = ['Test', 'Dev', 'Augmented Train']
        paths2 = ['exp/cem_data/test_beam5_inverted.pkl', 'exp/cem_data/preds/test_preds_simple256_logp.pkl']
        labels2 = ['Softmax Baseline', 'CEM 256 hidden units']
        paths3 = ['exp/cem_data/test_beam5_inverted.pkl']
        labels3 = ['Softmax Baseline']
        model = 'inv'
        plot(paths2, labels2, model)
