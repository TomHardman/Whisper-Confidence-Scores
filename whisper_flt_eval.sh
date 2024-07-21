export WANDB_API_KEY=28662375f12a901c5f50fa2109be8aa46bb9b015
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

echo $CUDA_VISIBLE_DEVICES

subset=flt_phrase_v2b
model=small.en

outdir=prompt0_lr1e-5_lower
echo $outdir


python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py \
    --outdir $outdir \
    --eval_list_file /scratches/dialfs/alta/sb2549/whisper_data/phrase_based/tsvs/flt/test_flt.tsv \
    --eval_dataset_name test \
    --stage evaluate \
    --model $model

#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_flt.tsv --cem_dataset_name test --stage gen_cem_data --model small.en
#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt_500.tsv --cem_dataset_name train_500 --stage gen_cem_data --model small.en --enable_specaug True --model_type dsf
#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  gec_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt_500.tsv --cem_dataset_name train_flt_500 --stage gen_cem_data --model small.en --enable_specaug True --max_time_warp 10 --max_freq_width 30 --n_freq_mask 2 --max_time_width 50 --n_time_mask 2 --model_type dsf
#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  gec_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/gec/train_gec_500.tsv --cem_dataset_name train_gec_500 --stage gen_cem_data --model small.en --enable_specaug True --max_time_warp 10 --max_freq_width 30 --n_freq_mask 2 --max_time_width 50 --n_time_mask 2
#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  gec_prompt0_lr1e-5_lower --cem_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/gec/dev_gec.tsv --cem_dataset_name dev_gec --stage gen_cem_data --model small.en
# python cem_predict.py --use_dec 0 --hidden_units 16 --batch_size 100 --ckpt_path /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_models/flt/train_beam5_SA10_30_2_50_2_inverted/16_pw2.0_usedec0/ckpt_epoch9_f0.44943_loss0.238_auc0.39169.pt --outfile exp/cem_preds/flt/Simple16_BCEpw2_usedec0.pkl