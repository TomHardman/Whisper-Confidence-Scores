from cem_model import SimpleCEM, ConfidenceDataset
import torch
from torch.utils.data import DataLoader
import argparse
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_confidence_scores(args):
    if args.use_dec:
        no_dec = False
    else:
        no_dec = True
    
    test_dataset = ConfidenceDataset(args.eval_file_path, no_dec=no_dec)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    confidence_model = SimpleCEM(no_dec=no_dec, n=args.hidden_units).to(device)
    saved_model = torch.load(args.ckpt_path)
    confidence_model.load_state_dict(saved_model['model_state_dict'])
    
    confidence_model.eval()
    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for i, (features, tgts) in enumerate(test_dataloader):
            # Compute prediction and loss
            features=features.to(device)
            tgts=tgts.to(device)
            pred = confidence_model(features)

            all_preds.extend([p.tolist()[0] for p in pred])
            all_tgts.extend([t.tolist()[0] for t in tgts])
    
    data = [all_preds, all_tgts]
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe wav files in a wav list')
    parser.add_argument('--ckpt_path', type=str, default='', help='path to the pretrained model')
    parser.add_argument('--eval_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/test_beam5_inverted.pkl', help='path to pickled test dataframe')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    parser.add_argument('--hidden_units', type=int, help='number of hidden units for simple CEM')
    parser.add_argument('--outfile', type=str, help='path to save prediction')
    parser.add_argument('--use_dec', type=int, default=0, help='whether to use decoder state in feature vector')
    args = parser.parse_args()
    
    data = predict_confidence_scores(args)
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
