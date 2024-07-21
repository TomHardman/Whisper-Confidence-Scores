import pickle
import argparse
import torch
import Levenshtein

#not_included = ["dominica's", "chihuahua", "flamboyant", "extracurricular"]
not_included = []

def create_word_conf_file(transcription_path, data_path, confs_file_path, outfile):
    phr_list_transcribe = []

    with open(transcription_path, 'r') as f:
        for line in f:
            id_ = line.split(':')[0]
            text = line.split(':')[1].split('\n')[0][1:]
            words = text.split(' ')
            if id_.split('-')[-1] == 'hyp':
                phr_list_transcribe.append((id_[:-4], words))
    
    with open(data_path, 'rb') as f:
        word_dicts = pickle.load(f)
    
    with open(confs_file_path, 'rb') as f:
        confs, labels = pickle.load(f)
        
    offset = 0
    phr_list_data = []
    conf_idx = 0
    word_idx = 0
    for id_, phr in phr_list_transcribe:
        phr_data = []
        phr_data.append((id_, 0, 0))
        null_count = 0
        for word in phr:
            if word_dicts[word_idx]['word'] == '':
                word_idx += 1
                conf_idx += 1
                null_count += 1

            elif word in not_included and word == word_dicts[word_idx]['word']:
                phr_data.append((word, 0, 0))
                word_idx += 1
                
            elif word == word_dicts[word_idx]['word']:
                phr_data.append((word, confs[conf_idx], labels[conf_idx]))
                word_idx += 1
                conf_idx += 1
            else:
                print('Misalignment Error', word, word_dicts[word_idx]['word'])
                phr_data.append((word, 0, 0))
        phr_list_data.append(phr_data)
        assert(len(phr) == len(phr_data) + null_count - 1)

    assert(len(phr_list_data) == len(phr_list_transcribe))
    
    with open (outfile, 'wb') as f:
        pickle.dump(phr_list_data, f)



def main(args):
    create_word_conf_file(args.transcription_path, args.data_path, args.confs_path, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for gec')
    parser.add_argument('--data_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/test_gec_beam5_layer6_data.pkl')
    parser.add_argument('--transcription_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/small.en/gec_prompt0_lr1e-5_lower/transcribe/True_test_gec_beam5_stampFalse_nonorm_layer6')
    parser.add_argument('--confs_path', type=str, default='exp/cem_models/gec/train_gec_beam5_SA10_30_2_50_2_layer6_data_word/16_pw1.0_no_dec_norm0_poolmax_probmax/ep21testf_0.4523_testAUC_0.4182_preds.pkl')
    parser.add_argument('--outfile', type=str, default='for_gec/cem_word_confidences_gec_test.pkl')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)

