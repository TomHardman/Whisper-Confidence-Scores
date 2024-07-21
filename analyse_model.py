import torch
import argparse
from cem_model import SimpleCEM
import matplotlib.pyplot as plt
import numpy as np

def analyse_model_weights(args):
    confidence_model = SimpleCEM(n=args.hidden_units, feature_mode=args.mode, norm=True)
    saved_model = torch.load(args.ckpt_path)
    confidence_model.load_state_dict(saved_model['model_state_dict'])
    fc_weights = confidence_model.fc_layer.weight
    magnitudes = torch.sqrt(torch.sum(fc_weights ** 2, dim=0)).tolist()
    
    plt.plot(magnitudes[:], label=r'Pr$(y_t)$'+f' Mean: {np.mean(magnitudes[2304:]):.3f}')
    plt.plot(magnitudes[:2304], label=r'Emb$(y_t)$'+f' Mean: {np.mean(magnitudes[1536:2304]):.3f}')
    plt.plot(magnitudes[:1536], label=r'$\boldsymbol{d_t}$'+f' Mean: {np.mean(magnitudes[768:1536]):.3f}')
    plt.plot(magnitudes[:768], label=r'$\boldsymbol{a_t}$'+f' Mean: {np.mean(magnitudes[:768]):.3f}')
    plt.legend()

    plt.ylabel('Sum of weights associated with input element')
    plt.xlabel('Index of input element')
    plt.savefig(args.outname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe wav files in a wav list')
    parser.add_argument('--ckpt_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_models/gec/train_gec_beam5_SA10_30_2_50_2_inverted/16_pw1.0_no_prob/epoch3_f0.33775_loss0.283_auc0.30125_UCE0.0660.pt', help='path to the trained model')
    parser.add_argument('--hidden_units', type=int, default=16, help='number of hidden units for simple CEM')
    parser.add_argument('--mode', type=str, required=True, help='decides which part of feature vector to use')
    parser.add_argument('--outname', type=str, required=True)
    args = parser.parse_args() 
    analyse_model_weights(args)