from cem_model import SimpleCEM, ConfidenceDataset
import torch
from torch.utils.data import DataLoader
import argparse
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_confidence_scores(args):
    test_dataset = ConfidenceDataset(args.eval_file_path, feature_mode=args.mode, pred_mode=args.pred_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    confidence_model = SimpleCEM(feature_mode=args.mode, n=args.hidden_units, norm=args.norm).to(device)
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
    parser.add_argument('--ckpt_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_models/flt/train_flt_beam5_SA5_22_2_0_0_layer-1_data_word/16_pw1.0_no_dec_norm0_poolmax_probmax/ep13_f0.43870_loss0.147_auc0.38867_UCE0.0876.pt', help='path to the pretrained model')
    parser.add_argument('--eval_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/dev_gec_beam5_layer6_data.pkl', help='path to pickled dev dataframe')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    parser.add_argument('--hidden_units', type=int, help='number of hidden units for simple CEM')
    parser.add_argument('--mode', type=str, required=True, help='Determines which parts of feature vector are to be fed to model: "emb_only", "no_dec", "no_prob", "all"')
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--pred_mode', type=str, required=True)
    args = parser.parse_args()
    
    data = predict_confidence_scores(args)
    preds_file = '.'.join(args.ckpt_path.split('.')[:-1]) + '_preds_dev.pkl'
    with open(preds_file, 'wb') as f:
        pickle.dump(data, f)