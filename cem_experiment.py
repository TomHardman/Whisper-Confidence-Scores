import subprocess
import argparse

hidden_units = [16, 32]
gammas = [1, 2, 5]
pos_weights = [1, 2]
use_dec = [0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment to train multiple CEMs')
    parser.add_argument('--train_path', type=str, default='exp/cem_data/train_gec_beam5_SA10_30_2_50_2_inverted.pkl')
    parser.add_argument('--dev_path', type=str, default='exp/cem_data/gec/dev_gec_beam5_inverted.pkl')
    parser.add_argument('--model', type=str, default='flt')
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    
    if args.mode == 'focal':
        for hu in hidden_units:
            for g in gammas:
                for ud in use_dec:
                    command = ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '0',
                               '--gamma', str(g), '--use_dec', str(ud), '--train_file_path', args.train_path,
                               '--dev_file_path', args.dev_path, '--whisper_model', args.model]
                    subprocess.run(command)
    
    elif args.mode == 'BCE':
        for hu in hidden_units:
            for pw in pos_weights:
                for ud in use_dec:
                    command = ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '1',
                                '--pos_weighting', str(pw), '--use_dec', str(ud), '--train_file_path', args.train_path,
                                '--dev_file_path', args.dev_path, '--whisper_model', args.model]
                    subprocess.run(command)
    