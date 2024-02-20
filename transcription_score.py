import editdistance
from alignment import token_align
import whisper
import argparse
import torch
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

def score_transcriptions(path, tokenizer, normalise_ref=False):
    w_errors = 0
    w_refs = 0
    token_errors = 0
    token_refs = 0
    wav_hyp = None
    wav_ref = None
    std = EnglishTextNormalizer()
    
    with open(path, 'r') as f:
        for line in tqdm(f):
            wav_path = line.split(':')[0]
            if wav_path.split('-')[-1] == 'hyp':
                hyp = line.split(': ')[1]
                hyp_tokens = tokenizer.encode(hyp)
                wav_hyp = wav_path[:-3]
            else:
                if normalise_ref:
                    ref = std(line.split(': ')[1])
                else:
                    ref = line.split(': ')[1]
                ref_tokens = tokenizer.encode(ref)
                wav_ref = wav_path[:-3]
            
            if wav_hyp == wav_ref:
                w_errors += editdistance.eval(hyp.split(), ref.split())
                w_refs += len(ref.split())

                token_truth_labels, incorrect = token_align(hyp_tokens, ref_tokens, tokenizer)
                token_errors += token_truth_labels.count(0)
                token_refs += len(ref_tokens)
        
        print(f'WER: {w_errors/w_refs}')
        print(f'Token error rate: {token_errors/token_refs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a transcription file')
    parser.add_argument('--path', type=str, help='Path to transcription file', default='exp/small.en/gec_prompt0_lr1e-5_lower/transcribe/True_test_gec_beam5_stampFalse_nonorm')
    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wmodel = whisper.load_model('small.en', device=args.device, download_root='/home/mifs/th624/.cache/whisper', dropout=0, ilm=False, lora=0)
    wtokenizer = whisper.tokenizer.get_tokenizer(False, language='en', task='transcribe', num_languages=wmodel.num_languages)

    score_transcriptions(args.path, wtokenizer)
