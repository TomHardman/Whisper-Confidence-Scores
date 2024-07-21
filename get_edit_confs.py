import errant
import argparse
import pickle
import os
import Levenshtein
import numpy as np
from tqdm import tqdm
# need to run this code in CU4_seq2seq environment


def get_stc_edits(annotator, src, tgt):
    src = annotator.parse(src)
    tgt = annotator.parse(tgt)
    edits = annotator.annotate(src, tgt)
    return edits


def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def preprocess_stc(words):
    first_word = True
    pp_words = []

    for word in words:
        word = word.lower()
        if first_word:
            word = word[0].upper() + word[1:]
        if word == 'i':
            word = 'I'
        elif word.split("'")[0] == 'i':
            word = 'I' +  ("'") + "'".join(word.split("'")[1:])
        pp_words.append(word)
        first_word = False

    pp_words = ' '.join(pp_words)    
    
    return pp_words


def equal_edits(edit_hyp, edit_ref, asr_shifts):
    if all([edit_hyp.o_start + asr_shifts[0][edit_hyp.o_start] == edit_ref.o_start, 
            edit_hyp.o_end + asr_shifts[1][edit_hyp.o_end] == edit_ref.o_end, 
            edit_hyp.c_str == edit_ref.c_str,
            edit_hyp.o_str == edit_ref.o_str]):
        return True
    return False


def main(args):
    with open(args.confs_path_flt, 'rb') as f:
        confs_flt = pickle.load(f)
    
    with open(args.confs_path_gec, 'rb') as f:
        confs_gec = pickle.load(f)
    
    stcs_trans = read_lines(args.trans_path)
    stcs_ref = read_lines(args.ref_path)

    annotator = errant.load('en')
    ref_tot = 0
    tp = 0
    fp = 0

    edits_hyp = []
    edits_fn = []

    for stc_trans, stc_ref, stc_data_flt, stc_data_gec in tqdm(zip(stcs_trans, stcs_ref, confs_flt, confs_gec), total=len(stcs_trans)):

        stc_asr = [word[0] for word in stc_data_flt]
        stc_asr_confs = [1 - conf[1] for conf in stc_data_flt]
        stc_asr_labels = [int(abs(label[2] - 1)) for label in stc_data_flt]
        
        stc_pred = [word[0] for word in stc_data_gec]
        stc_pred_confs = [1 - conf[1] for conf in stc_data_gec]
        stc_pred_labels = [int(abs(label[2] - 1)) for label in stc_data_gec]

        stc_asr, stc_pred = preprocess_stc(stc_asr), preprocess_stc(stc_pred)

        hyp_edits = get_stc_edits(annotator, stc_asr, stc_pred)
        ref_edits = get_stc_edits(annotator, stc_trans, stc_ref)
        ref_tot += len(ref_edits)

        # find operations to get from transcription to asr such that we can manually adjust spans of edits to match
        asr_editops = Levenshtein.opcodes(stc_asr.split(' '), stc_trans.split(' '))
        asr_shifts = np.zeros((2, len(stc_asr.split(' ')) + 2))
        
        for opcode, asr_start, asr_stop, trans_start, trans_stop in asr_editops:
            if opcode == 'insert':
                asr_shifts[0, asr_start] += trans_stop - trans_start
                asr_shifts[1, asr_start+1] += trans_stop - trans_start
            elif opcode == 'delete':
                asr_shifts[:, asr_stop] += - (asr_stop - asr_start)
        
        asr_shifts = np.cumsum(np.array(asr_shifts), axis=1)

        for hyp_edit in hyp_edits:
            edit_label = 0
            for ref_edit in ref_edits:
                if equal_edits(hyp_edit, ref_edit, asr_shifts=asr_shifts):
                    tp += 1
                    edit_label = 1
                    ref_edits.remove(ref_edit)
                    break
            if not edit_label:
                fp += 1
            
            edit = (edit_label, stc_asr_confs[hyp_edit.o_start:hyp_edit.o_end], stc_pred_confs[hyp_edit.c_start:hyp_edit.c_end], hyp_edit.type)
            edits_hyp.append(edit)
    
        for ref_edit in ref_edits:
            edit = (0, ref_edit.type)
            edits_fn.append(edit)
    
    fn = len(edits_fn)
    print(f'P: {tp/(tp +fp)}, R: {tp/(tp+fn)}')

    with open(args.outfile_hyp, 'wb') as f:
        pickle.dump(edits_hyp, f)
    
    with open(args.outfile_fn, 'wb') as f:
        pickle.dump(edits_fn, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for gec')
    parser.add_argument('--confs_path_flt', type=str, default='for_gec/cem_word_confidences_flt_test_old.pkl')
    parser.add_argument('--confs_path_gec', type=str, default='for_gec/cem_word_confidences_gec_test.pkl')
    parser.add_argument('--ref_path', type=str, default='../data/Linguaskill/preprocessed/ls_test.corr')
    parser.add_argument('--trans_path', type=str, default='../data/Linguaskill/preprocessed/ls_test.inc')
    parser.add_argument('--outfile_hyp', type=str, default='for_gec/edit_data_hyp_test.pkl')
    parser.add_argument('--outfile_fn', type=str, default='for_gec/edit_data_fn_test.pkl')
    args = parser.parse_args()
    main(args)