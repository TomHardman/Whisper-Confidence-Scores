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
import csv
from collections import defaultdict
from tqdm import tqdm


def csv_to_pickle(path, toy=False):
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        if toy:
            print('Toy')
            csv_reader_temp = []
            i = 0
            for row in csv_reader:
                if i < 5000:
                    csv_reader_temp.append(row)
                    i += 1
                else:
                    break
            print(len(csv_reader_temp))
            csv_reader = csv_reader_temp

    
        data_list = []
        
        for row in tqdm(csv_reader):
            word_dict = defaultdict(list)
            word_dict['word'] = row[0]
            i = 1
            while True:
                try:
                    token_data = row[i: i+2307]
                    word_dict['tokens'].append(token_data[0])
                    word_dict['attn_features'].append(np.array(token_data[1:769]).astype('float16'))
                    word_dict['dec_features'].append(np.array(token_data[769:1537]).astype('float16'))
                    word_dict['emb'].append(np.array(token_data[1537:2305]).astype('float16'))
                    word_dict['sm_probs'].append(1 - float(token_data[-2])) # invert labels and probabilities as we want to detect incorrectness
                    word_dict['labels'].append(abs(int(float(token_data[-1]) - 1)))
                    i += 2307
                
                except IndexError:
                    break
        
            data_list.append(word_dict)

    # Save as pickle file
    toy_str = '_toy' if toy else ''
    pickle_file = path.split('.')[0] + toy_str + '_data.pkl'
        
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_list, f)


def create_distorted_data(pickle_path):
    print('creating distorted data')

    with open(pickle_path, 'rb') as f:
        word_dicts = pickle.load(f)

    new_data = []

    neg_count = 0

    for wd in tqdm(word_dicts):
        if wd['labels'][0] == 1:
            new_data.append(wd)
            neg_count += 1
        
        elif wd['labels'][0] == 0 and neg_count:
            new_data.append(wd)
            neg_count -= 1
    
    new_path = pickle_path + '_distorted'

    with open(new_path, 'wb') as f:
        pickle.dump(new_data, f)
            


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
    return p, r, t


def f_score(p, r, beta=0.5):
    num = 1+beta**2
    denom = 1/p + beta**2/r
    return num/denom

def plot_pr_curve(p, r, t, fig, ax, label, first_plot, show_stats):
    f = f_score(np.array(p), np.array(r))
    best_f = np.max(f)
    t_at_bf = t[np.argmax(f)]
    if first_plot:
        ax.scatter(r[np.argmax(f)], p[np.argmax(f)], color='red', marker='x', zorder=3,
                label=r'Best $F_{0.5}$')
    else:
        ax.scatter(r[np.argmax(f)], p[np.argmax(f)], color='red', marker='x', zorder=3)
    
    if show_stats:
        ax.plot(r, p, label=f'{label} AUC: {auc(r, p):.4f}, Best F: {best_f:.4f}')
    
    else:
        ax.plot(r, p, label=f'{label}')


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


def plot_reliability_diagram(confs, y, label, model, n=20, plot=True, show_stats=True):
    title = label + ' ' + model

    confs = 1 - np.array(confs)
    y = abs(1 - np.array(y))
    
    ECE, UCE, accs, count = rel_diagram(confs, y, n)

    if plot:
        fig, ax = plt.subplots()
        if show_stats:
            ax.bar(accs, count, label=f'ECE: {ECE:.5f}, UCE: {UCE:.5f}', width=accs[1]-accs[0], alpha=0.8)
        else:
            ax.bar(accs, count, width=accs[1]-accs[0], alpha=0.8)
        
        ax.plot(accs, accs, '--', color='orange', label='perfect calibration') # calibration line
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        #ax.set_title(label)
        ax.legend()
        fig.savefig(f'exp/rel_diagrams/{title.replace(" ", "_")}')
    
    return ECE, UCE


def plot_token_combs(pickle_path, cem_preds_path=None, show_stats=True,
                     save_cem_preds=True):
     with open(pickle_path, 'rb') as f:
        word_dicts = pickle.load(f)
        inv_prod_probs = []
        prod_probs = []
        max_probs = []

        if cem_preds_path:
            inv_prod_probs_cem = []
            prod_probs_cem = []
            max_probs_cem = []
            with open(cem_preds_path, 'rb') as f:
                cem_preds = pickle.load(f)[0]
            pointer = 0
            print(len(word_dicts), len(cem_preds))
        
        targets = []
        labels = [('1 - Prod Incorrect', '1 - Prod Incorrect'),
                  ('Min', 'Min'),
                  ('Prod Correct', 'Prod Correct')]

        for wd in word_dicts:
            probs = np.array(wd['sm_probs'])
            inv_probs = 1 - probs # prob of correctness

            max_probs.append(np.max(probs))
            prod_probs.append(np.exp(np.sum(np.log(probs))))
            inv_prod_probs.append(1 - np.exp(np.sum(np.log(inv_probs))))
            targets.append(wd['labels'][0])

            if cem_preds_path:
                n_tokens = len(probs) # number of tokens in current word
                cem_probs = np.array(cem_preds[pointer: pointer+n_tokens]) # extract predicted confidence scores for tokens corresponding to word
                cem_inv_probs = 1 - cem_probs
                max_probs_cem.append(np.max(cem_probs))
                prod_probs_cem.append(np.exp(np.sum(np.log(cem_probs))))
                inv_prod_probs_cem.append(1 - np.exp(np.sum(np.log(cem_inv_probs))))
                pointer += n_tokens
        
        fig, ax = plt.subplots()
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        if cem_preds_path:
            first_plot = True
            for confs, label in zip([prod_probs_cem, max_probs_cem, inv_prod_probs_cem], labels):
                p, r, t = pr_curve(targets, confs)
                plot_pr_curve(p, r, t, fig, ax, label[0], first_plot=first_plot, show_stats=show_stats)
                rel_label = 'CEM trained on tokens ' + label[1] 
                plot_reliability_diagram(confs, targets, rel_label, model, n=20, plot=True, 
                                         show_stats=show_stats)
                first_plot = False
        
        ax.legend(loc='upper right')         
        fig.savefig(f'exp/pr_curves/pr_token_cem_{model}.png')

        fig, ax = plt.subplots()
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        first_plot = True
        for confs, label in zip([prod_probs, max_probs, inv_prod_probs], labels):
            if label[1] == 'Prod Correct':
                p, r, t = pr_curve(targets, confs)
                plot_pr_curve(p, r, t, fig, ax, 'Softmax Baseline', first_plot=first_plot, 
                            show_stats=show_stats)
                rel_label = 'Softmax Baseline ' + label[1] 
                plot_reliability_diagram(confs, targets, label[0], model, n=20, plot=True,
                                        show_stats=show_stats)
                first_plot = False
        
        ax.legend(loc='upper right')         
        fig.savefig(f'exp/pr_curves/pr_softmax_baseline_{model}.png')
        
        if cem_preds_path and save_cem_preds:
            fp = '.'.join(cem_preds_path.split('.')[:-1]) + '_word.pkl'
            
            with open(fp, 'wb') as f:
                pickle.dump((max_probs_cem, targets), f)

        with open('exp/cem_data/flt/softmax_preds.pkl', 'wb') as f:
            pickle.dump((inv_prod_probs, targets), f)


def plot_predictions(paths, show_stats, labels=None):
    fig, ax = plt.subplots()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    labels = ['Softmax Baseline',
              r'CEM']

    first_plot=True
    
    for path, label in zip(paths, labels):
        with open(path, 'rb') as f:
            confs, tgts = pickle.load(f)
        
        print(len(paths))
        
        p, r, t = pr_curve(tgts, confs)
        plot_pr_curve(p, r, t, fig, ax, label, first_plot, show_stats=show_stats)
        rel_label = label
        plot_reliability_diagram(confs, tgts, rel_label, model, n=20, plot=True,
                                 show_stats=show_stats)
        first_plot=False
    
    ax.legend(loc='upper right')         
    fig.savefig(f'exp/pr_curves/pr_word_cem_{model}.png')


def hist_plot(path):
    with open(path, 'rb') as f:
            word_dicts = pickle.load(f)

    confs = []
    y = []

    for wd in word_dicts:
        probs = 1 - np.array(wd['sm_probs'])
        confs.append(np.exp(np.sum(np.log(probs))))
        y.append(wd['labels'][0])

    confs = 1 - np.array(confs)

    # Separate confidences based on target values
    confidences_target0 = [1 - confs[i] for i in range(len(confs)) if y[i]==0]
    confidences_target1 = [1 - confs[i] for i in range(len(confs)) if y[i]==1]

    # Plot histograms with different colors for each target
    plt.hist([confidences_target0, confidences_target1], bins=10, color=['blue', 'red'], label=['Correct', 'Incorrect'])
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
    parser.add_argument('--mode', type=str, default='plot_token_baseline')
    parser.add_argument('--filepath', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/flt/test_flt_beam5_layer-1_data.pkl')
    parser.add_argument('--pred_path_token', type=str, help='Path to token based CEM prediction file',
                        default=None)
    parser.add_argument('--pred_paths', type=list_type, default='exp/cem_data/flt/softmax_preds.pkl,exp/cem_models/flt/train_flt_beam5_SA5_22_2_0_0_layer-1_data_word/16_pw1.0_no_dec_norm0_poolmax_probmax/ep13testf_0.4505_testAUC_0.4003_preds.pkl')
    parser.add_argument('--labels', type=label_list_type, default='Word_CEM_Product_Agg,Word_CEM_Max_Agg,Word_CEM_Inv_Prod_Agg')
    parser.add_argument('--only_baseline', default=0)
    parser.add_argument('--show_stats', default=1, type=int)
    args = parser.parse_args()

    model_names = {'gec': 'Whisper GEC',
                   'flt': 'Whisper Fluent',
                   'dsf': 'Whisper Disfluent'}

    if args.mode == 'convert':
        csv_path = args.filepath
        csv_to_pickle(csv_path)
    
    elif args.mode == 'plot_token_baseline':
        model = model_names.get(args.model)
        plot_token_combs(args.filepath, cem_preds_path=args.pred_path_token, 
                         show_stats=args.show_stats)
    
    elif args.mode =='distorted_data':
        pickle_path = args.filepath
        create_distorted_data(pickle_path)
    
    elif args.mode == 'hist':
        hist_plot(args.filepath)

    else:
        model = model_names.get(args.model)
        plot_predictions(args.pred_paths, args.show_stats)
        
