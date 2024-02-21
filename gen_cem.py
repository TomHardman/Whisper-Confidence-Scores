import subprocess

specaug_params = {
    'W': 5,
    'F': 22,
    'mF': 2,
    'T': 0,
    'mt': 0,

}

cmd_train = f'python3 whisper_finetune.py --outdir prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt.tsv --cem_dataset_name train --stage gen_cem_data --model small.en --enable_specaug True'
cmd_dev = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/dev_flt.tsv --cem_dataset_name dev --stage gen_cem_data --model small.en --enable_specaug False'
cmd_test = 'python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_flt.tsv --cem_dataset_name test --stage gen_cem_data --model small.en --enable_specaug False'

if __name__ == '__main__':
    for cmd in [cmd_test]:
        cmd = cmd.split(' ')
        print(cmd)
        subprocess.run(cmd)