import subprocess
import argparse

hidden_units = [16]
feature_modes = ['no_dec']
fm_tokens = ['no_dec']
pool_modes = ['max']
agg_modes = ['max']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment to train multiple CEMs')
    parser.add_argument('--train_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/train_gec_beam5_SA20_50_2_50_2_layer-1_data.pkl')
    parser.add_argument('--dev_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/dev_gec_beam5_layer-1_data.pkl')
    parser.add_argument('--test_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/gec/test_gec_beam5_layer-1_data.pkl')
    parser.add_argument('--model', type=str, default='gec')
    parser.add_argument('--loss', type=str, default='BCE')
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--prob_skip', type=int, default=0)
    parser.add_argument('--pred_mode', type=str, default='word_new')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--deep', type=int, default=0)
    args = parser.parse_args()
    
    
    if args.pred_mode == 'word':
        for hu in hidden_units:
            for pm in pool_modes:
                for am in agg_modes:
                    for fm in feature_modes:
                        command = ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '1',
                                    '--pos_weighting', '1.0', '--feature_mode', fm, '--train_file_path', args.train_path,
                                    '--dev_file_path', args.dev_path, '--test_file_path', args.test_path, '--whisper_model', args.model,
                                    '--norm', str(args.norm), '--prob_skip', str(args.prob_skip), '--agg_mode', am,
                                    '--pool_mode', pm, '--pred_mode', args.pred_mode, '--batch_size', str(args.batch_size),
                                    '--deep', str(args.deep)]
                        subprocess.run(command)
    
    if args.pred_mode == 'token':
        for hu in hu_tokens:
            for fm in fm_tokens:
                command = ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '1',
                            '--pos_weighting', '1.0', '--feature_mode', fm, '--train_file_path', args.train_path,
                            '--dev_file_path', args.dev_path, '--test_file_path', args.test_path, '--whisper_model', args.model,
                            '--norm', str(args.norm), '--prob_skip', str(args.prob_skip), '--agg_mode', 'max',
                            '--pool_mode', 'max', '--pred_mode', 'token', '--deep', str(args.deep)]
                subprocess.run(command)
    
    if args.pred_mode == 'word_new':
        print('Starting')
        for hu in hidden_units:
            for fm in feature_modes:
                command = ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '1',
                            '--pos_weighting', '1.0', '--feature_mode', fm, '--train_file_path', args.train_path,
                            '--dev_file_path', args.dev_path, '--test_file_path', args.test_path, '--whisper_model', args.model,
                            '--norm', str(args.norm), '--prob_skip', str(args.prob_skip), '--pred_mode', args.pred_mode, 
                            '--batch_size', str(args.batch_size), '--deep', str(args.deep)]
                subprocess.run(command)
    
    if args.pred_mode == 'all':
        for pred_mode in ['token', 'word', 'word_new']:
            for hu in hidden_units:
                command =  ['python', 'cem_train.py', '--hidden_units', str(hu), '--BCE_loss', '1',
                            '--pos_weighting', '1.0', '--feature_mode', 'no_dec', '--train_file_path', args.train_path,
                            '--dev_file_path', args.dev_path, '--test_file_path', args.test_path, '--whisper_model', args.model,
                            '--norm', str(args.norm), '--prob_skip', str(args.prob_skip), '--agg_mode', 'max',
                            '--pool_mode', 'max', '--pred_mode', pred_mode, '--batch_size', str(args.batch_size),
                            '--deep', str(args.deep)]
                subprocess.run(command)

    