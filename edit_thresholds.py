import argparse
import pickle
import numpy as np
from tools import f_score
from sklearn.metrics import precision_recall_curve as pr_curve, auc
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_pr_curve(p, r, t, fig, ax, label):
    f = f_score(np.array(p), np.array(r))
    best_f = np.max(f)
    idx_opt = np.argmax(f)
    ax.plot(r, p, label=f'{label}, Best F: {best_f:.4f} at P: {p[idx_opt]:.4f}, R: {r[idx_opt]:.4f}')
    #ax.plot(r, p, label=f'{label}')


def get_edit_conf(probs_flt, probs_gec, mode):
    if mode == 1:
            conf = 1
            probs = probs_flt + probs_gec
            for prob in probs:
                conf *= prob
        
    elif mode == 2:
        conf = 1
        probs = probs_gec
        for prob in probs:
            conf *= prob
    
    elif mode == 3:
        probs = probs_flt + probs_gec
        H = 0
        for prob in probs:
            H -= prob * np.log(prob)
        conf = - H
    
    elif mode == 4:
        probs = probs_flt + probs_gec
        conf = max(probs)
    
    elif mode == 5:
        max1 = max(probs_flt) if probs_flt else 1
        max2 = max(probs_gec) if probs_gec else 2
        conf = max1 * max2

    elif mode == 6:
        probs = probs_flt + probs_gec
        conf = min(probs)

    elif mode == 7:
        min1 = min(probs_flt) if probs_flt else 1
        min2 = min(probs_gec) if probs_gec else 2
        conf = min1 * min2

    elif mode == 8:
        probs = probs_flt + probs_gec
        conf = np.mean(probs)
    
    elif mode == 9:
        probs = probs_flt + probs_gec
        conf = np.sqrt(np.prod(probs))**len(probs)
    
    elif mode == 10:
        conf = 1
        probs = probs_flt
        for prob in probs:
            conf *= prob
    
    elif mode.mode == 11:
        probs = probs_flt
        conf = np.mean(probs) if probs else 1
    
    return conf


def select_type(edit_type, edits_hyp, edits_ref):
    edits_specific_hyp = []
    hyp_idxs = []
    edits_specific_ref = []

    for i, edit in enumerate(edits_hyp):
        typ = edit[3]
        if typ == edit_type:
            edits_specific_hyp.append(edit)
            hyp_idxs.append(i)
    
    for edit in edits_ref:
        typ = edit[1]
        if typ == edit_type:
            edits_specific_ref.append(edit)
    
    return edits_specific_hyp, edits_specific_ref, hyp_idxs


def get_all_types(edits_hyp, count=False):
    if not count:
        types = set()

        for edit in edits_hyp:
            typ = edit[3]
            if typ not in types:
                types.add(typ)
    
    else:
        types = defaultdict(int)
        for edit in edits_hyp:
            typ = edit[3]
            types[typ] += 1
    
    return types


def analyse_edits(edits_hyp, edits_fn, n=10):
    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)
    counts_dict_gec = defaultdict(int)
    counts_dict_manual = defaultdict(int)

    for edit in edits_hyp:
        if edit[0] == 1:
            tp_dict[edit[3]] += 1
            counts_dict_manual[edit[3]] += 1
        else:
            fp_dict[edit[3]] += 1
        counts_dict_gec[edit[3]] += 1

    for edit in edits_fn:
        fn_dict[edit[1]] += 1
        counts_dict_manual[edit[1]] += 1

    top_types = []
    
    for item in sorted(counts_dict_manual.items(), key = lambda x:x[1], reverse=True)[:n]:
        edit_type = item[0]
        top_types.append(edit_type)
        
        tp = tp_dict[edit_type]
        fp = fp_dict[edit_type]
        fn = fn_dict[edit_type]
        p = tp/(tp+fp)
        r = tp/(tp + fn)
        f = f_score(p, r)

        print(f'{edit_type}: Precision: {p:.4f},  Recall: {r:.4f}, F0.5: {f:.4f}')
    
    return top_types
    

def get_stats_by_type(edits_hyp, edits_fn, confs, labels, types, t):
    for typ in types:
        edits_hyp_typ, edits_fn_typ, idxs = select_type(typ, edits_hyp, edits_fn)
        confs_typ = [confs[idx] for idx in idxs]
        labels_typ = [labels[idx] for idx in idxs]

        tp = 0
        fp_ = 0
        x = 0

        for conf, label in zip(confs_typ, labels_typ):
            if label == 1:
                tp += 1
                if conf < t:
                    x += 1
            
            elif label == 0 and conf >= t:
                fp_ += 1
            
        fn = len(edits_fn_typ)
        p = (tp - x)/(tp - x + fp_)
        r = (tp - x)/(tp + fn)
        f = f_score(p, r)
        
        print(f'{typ}: Precision: {p:.4f},  Recall: {r:.4f}, F0.5: {f:.4f}')

    
def get_pr(edits_hyp, edits_fn, mode, edit_type=None,
           stats_by_type=False, types=None, return_basic=False):
    
    if edit_type:
        edits_hyp, edits_fn, _ = select_type(edit_type, edits_hyp, edits_fn)

    labels = []
    confs = []

    for edit in edits_hyp:
        labels.append(edit[0])
        probs_flt = edit[1]
        probs_gec = edit[2]
        conf = get_edit_conf(probs_flt, probs_gec, mode)

        confs.append(conf)
    
    p, r, t = pr_curve(labels, confs)
    
    tp = sum(labels)
    fn = len(edits_fn)
    x = (tp - tp*r).astype(int)
    
    r_actual = (tp - x)/(tp + fn)
    f = f_score(p, r_actual)

    best_f, idx = max(f), np.argmax(f)
    best_t = t[idx]

    if stats_by_type and not edit_type:
        get_stats_by_type(edits_hyp, edits_fn, confs,
                          labels, types, best_t)
        
    if return_basic:
       tp_ = tp - x[idx]
       fn_ = fn + x[idx]
       fp_ = ((tp_ - p[idx] * tp_)/p[idx]).astype(int)
       if fp_ == np.nan or fp_ < 0:
            fp_ = 0
       return tp_, fp_, fn_

    return p, r_actual, f, t


def optimal_performance(edits_hyp_test, edits_fn_test, edits_hyp_dev=None, edits_fn_dev=None, mode=8,
                        cutoff=3, best_global=True):
    if edits_hyp_dev and edits_fn_dev:
        types = get_all_types(edits_hyp_dev)
        types_counts = get_all_types(edits_hyp_dev, count=True)
        op_point_by_type = defaultdict(float)
        p, r, f, t = get_pr(edits_hyp_dev, edits_fn_dev, mode)
        t_glob = t[np.argmax(f)]

        for typ in types:
            p, r, f, t = get_pr(edits_hyp_dev, edits_fn_dev, mode, typ)
            op_point_by_type[typ] = t[np.argmax(f)]
        
        tp = 0
        fp = 0
        x = 0
        bad_types = []

        for edit in edits_hyp_test:
            typ = edit[3] 
            
            if typ in op_point_by_type and types_counts[typ] > cutoff and not best_global:
                thr = op_point_by_type[typ]
            else:
                thr = t_glob
            
            lab = edit[0]
            probs_flt = edit[1]
            probs_gec = edit[2]
            conf = get_edit_conf(probs_flt, probs_gec, mode)

            if lab == 1:
                if conf >= thr and typ not in bad_types:
                    tp += 1
                else:
                    x += 1
            elif lab == 0 and conf >= thr and typ not in bad_types:
                fp += 1
        
        fn = len(edits_fn_test) + x

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = f_score(p, r)

        if best_global:
            print(f'Global best operating point: Precision: {p:.4f},  Recall: {r:.4f}, F0.5: {f:.4f}')
        else:
            print(f'Type-based best operating point: Precision: {p:.4f},  Recall: {r:.4f}, F0.5: {f:.4f}')


    else:
        types = get_all_types(edits_hyp_test)
        op_point_by_type = defaultdict(float)
        tp = 0
        fp = 0
        fn = 0

        for typ in types:
            tp_typ, fp_typ, fn_typ = get_pr(edits_hyp_test, edits_fn_test, mode, typ, return_basic=True)
            tp += tp_typ
            fp += fp_typ
            fn += fn_typ
        
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = f_score(p, r)
        print(f'Type-based best operating point: {typ}: Precision: {p:.4f},  Recall: {r:.4f}, F0.5: {f:.4f}')

            

modes = [1, 2, 10, 8, 9]
labels = ['Prod. All', 'Prod. GEC', 'Prod. Flt', 'Arith. Mean',
          'Geo. Mean']

def main(args):
    with open(args.path_hyp_test, 'rb') as f:
        edits_hyp_test = pickle.load(f)
    
    with open(args.path_fn_test, 'rb') as f:
        edits_fn_test = pickle.load(f)
    
    with open(args.path_hyp_dev, 'rb') as f:
        edits_hyp_dev = pickle.load(f)
    
    with open(args.path_fn_dev, 'rb') as f:
        edits_fn_dev = pickle.load(f)
    
    fig, ax = plt.subplots()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    for mode, label in zip(modes, labels):
        print('-------Analysing Edits------')
        print('Most common edits:')
        types = analyse_edits(edits_hyp_test, edits_fn_test)
        print('---------------------')

        print('----Analysing Common Edits at Best Global Operating Point----')
        p, r, f, t = get_pr(edits_hyp_test, edits_fn_test, mode=mode, edit_type=None,
                            stats_by_type=True, types=types)
        print(p[0], r[0], f[0])
        plot_pr_curve(p, r, t, fig, ax, label)
        print('-------------')
        
        print('------Analysing Edits at Best Operating Point by Type-----')
        for typ in types:
            p, r, f, t = get_pr(edits_hyp_test, edits_fn_test, mode=mode, edit_type=typ)
            idx = np.argmax(f)
            print(f'{typ}: Precision: {p[idx]:.4f},  Recall: {r[idx]:.4f}, F0.5: {f[idx]:.4f}')
        
        print('-------- Optimal performance: -------')
        #optimal_performance(edits_hyp_test, edits_fn_test)
        optimal_performance(edits_hyp_test, edits_fn_test, edits_hyp_dev, edits_fn_dev,
                            best_global=True)
        optimal_performance(edits_hyp_test, edits_fn_test, edits_hyp_dev, edits_fn_dev,
                            best_global=False)
        print('-----------------------')
    
    ax.legend(loc='upper right')         
    fig.savefig(f'exp/pr_curves_old/whisper_gec_edits/all_methods.png')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for gec')
    parser.add_argument('--path_hyp_test', type=str, default='for_gec/edit_data_hyp_test.pkl')
    parser.add_argument('--path_fn_test', type=str, default='for_gec/edit_data_fn_test.pkl')
    parser.add_argument('--path_hyp_dev', type=str, default='for_gec/edit_data_hyp_dev.pkl')
    parser.add_argument('--path_fn_dev', type=str, default='for_gec/edit_data_fn_dev.pkl')
    mode = parser.parse_args()
    main(mode)