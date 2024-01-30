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

#python3 /scratches/dialfs/alta/th624/exp-th624/Whisper_flt/whisper_finetune.py --outdir  prompt0_lr1e-5_lower --eval_list_file /scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_single_flt.tsv --eval_dataset_name test --stage gen_cem_data --model small.en