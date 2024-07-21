import subprocess

cmd_train_flt2_1 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt.tsv --cem_dataset_name train_flt_ --stage gen_cem_data --model small.en --enable_specaug True --model_type flt --max_time_warp 10 --max_freq_width 30 --n_freq_mask 2 --max_time_width 50 --n_time_mask 2 --cem_layer 1'
cmd_train_flt2_6 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt.tsv --cem_dataset_name train_flt_ --stage gen_cem_data --model small.en --enable_specaug True --model_type flt --max_time_warp 10 --max_freq_width 30 --n_freq_mask 2 --max_time_width 50 --n_time_mask 2 --cem_layer 6'
cmd_train_flt1_6 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt.tsv --cem_dataset_name train_flt_ --stage gen_cem_data --model small.en --enable_specaug True --model_type flt --max_time_warp 5 --max_freq_width 22 --n_freq_mask 2 --max_time_width 0 --n_time_mask 0 --cem_layer 6'



cmd_dev_flt_6 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/dev_flt.tsv --cem_dataset_name dev_flt --stage gen_cem_data --model small.en --enable_specaug False --model_type flt --cem_layer 6'
cmd_test_flt_6 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_flt.tsv --cem_dataset_name test_flt --stage gen_cem_data --model small.en --enable_specaug False --model_type flt --cem_layer 6'
cmd_dev_flt_1 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/dev_flt.tsv --cem_dataset_name dev_flt --stage gen_cem_data --model small.en --enable_specaug False --model_type flt --cem_layer 1'
cmd_test_flt_1 = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir flt_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_flt.tsv --cem_dataset_name test_flt --stage gen_cem_data --model small.en --enable_specaug False --model_type flt --cem_layer 1'

if __name__ == '__main__':
    for cmd in [cmd_train_flt1_6]:
            cmd_s = cmd.split(' ')
            subprocess.run(cmd_s)