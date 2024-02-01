#!/usr/bin/python3
# Finetune Whisper ASR
# Modified from https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz#scrollTo=4lrkSd9tjF8X
# Rao Ma, 2023-03-20
import argparse
import evaluate as heval
from general import save_script_args, check_output_dir, str2bool
from collections import defaultdict
import wandb
import pdb
import os
import numpy as np
import torch
from torch import nn
import editdistance
from collections import defaultdict
import torchaudio
import torchaudio.transforms as at
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from collections import Counter
import numpy
import whisper
import csv
print(whisper)
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer, EnglishNumberNormalizer
from tqdm import tqdm
import pdb
import collections
import math
import random
import matplotlib.pyplot as plt
import re
import loralib
import time
from whisper.audio import SAMPLE_RATE, TOKENS_PER_SECOND
from alignment import token_align
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

def load_wave(wave_path, sample_rate:int=16000, st_time:int=-1, ed_time:int=-1) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if st_time >= 0 and ed_time >= 0:
        waveform = waveform[st_time : ed_time]
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def load_audio_file_list(pair_list_path, ilm=False):
    audio_transcript_pair_list = []
    audio_id = None
    for line in open(pair_list_path):
        line_sp = line.strip().split(None, 2)
        if not audio_id:
            audio_id = line_sp[0]
        elif audio_id != line_sp[0] and ilm:
            break
        if len(line_sp) == 2:
            line_sp.append('')
        assert len(line_sp) == 3
        audio_transcript_pair_list.append(line_sp)
    return audio_transcript_pair_list

# Text format conversion
def text_convert(text, norm='lower'):
    if norm == 'lower':
        text = text.lower()
    elif norm == 'upper':
        text = text.upper()
    elif norm == 'capitalize':
        text = text.capitalize()
    else:
        text = text
    return text

def init_embedding(model, tokenizer, n_decoder_prompts, device, scheme=None, audio_file_list=None):
    if not scheme:
        return
    elif scheme == 'random':
        valid_list = []
        for ii in range(tokenizer.eot):
            if re.search(r'[^a-z ]', tokenizer.decode([ii])) is None:
                valid_list.append(ii)
        inds = random.sample(valid_list, n_decoder_prompts)
    elif scheme == 'top':
        ind_list = []
        for line in open(audio_file_list):
            text = text_convert(line.strip().split(None, 2)[-1])
            ind_list.extend(tokenizer.encode(text))
        count = Counter(ind_list)
        inds = [x[0] for x in count.most_common(n_decoder_prompts)]
    else:
        assert 0

    print(scheme, inds)
    for ind in inds:
        print(tokenizer.decode([ind]))
    tensors = torch.LongTensor(inds).to(device)
    embs = model.decoder.token_embedding(tensors)
    model.decoder.learned_embedding.data.copy_(embs.detach())
    print('====== Init embeddings ======')


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, args, text_norm='str', ref_file=None, audio=None, init_prompt=None, few_shot_lst=None, fix_tokenizer=False, n_mels=80) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.text_norm = text_norm
        self.args = args
        self.audio = audio
        self.init_prompt = init_prompt
        self.few_shot_lst = few_shot_lst
        self.sep = ' '
        self.fix_tokenizer = fix_tokenizer
        self.n_mels = n_mels

        self.reference_dict = {}
        if ref_file:
            for line in open(ref_file):
                line = line.strip()
                if '-hyp' in line:
                    audio_id = line.split()[0][:-5]
                    if len(line.split()) == 1:
                        text = ''
                    else:
                        text = line.split(None, 1)[1]
                elif '-ref' in line:
                    pass
                else:
                    audio_id = line.split()[0]
                    text = line.split(None, 2)[-1]
                self.reference_dict[audio_id] = text.lower()

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]
        if text.startswith('st:'):
            st_time, text = text.split(None, 1)
            ed_time, text = text.split(None, 1)
            st_time = int(float(st_time[3:]) * SAMPLE_RATE)
            ed_time = int(float(ed_time[3:]) * SAMPLE_RATE)
        else:
            st_time, ed_time = -1, -1
        # audio
        if self.audio is not None:
            audio = self.audio
        else:
            audio = load_wave(audio_path, sample_rate=self.sample_rate, st_time=st_time, ed_time=ed_time)

        if self.few_shot_lst:
            print(audio.shape)
            wav_lst = []
            for _, exp_audio_path, _ in self.few_shot_lst:
                exp_audio = load_wave(audio_path, sample_rate=self.sample_rate)
                wav_lst.append(exp_audio)
            wav_lst.append(audio)
            audio = torch.cat(wav_lst, dim=1)
            # audio = torch.zeros_like(audio)

        audio = whisper.pad_or_trim(audio.flatten(), length=args.chunk_size * SAMPLE_RATE)
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)
        if self.args.enable_specaug:
            mel = whisper.spec_augment(mel, max_time_warp=self.args.max_time_warp, max_freq_width=self.args.max_freq_width, n_freq_mask=self.args.n_freq_mask, max_time_width=self.args.max_time_width, n_time_mask=self.args.n_time_mask)

        st_pos = 0
        if self.few_shot_lst:
            text_lst = []
            for _, _, exp_text in self.few_shot_lst:
                text_lst.append(exp_text.strip())
            prefix_text = self.sep.join(text_lst)           # TODO: change delimeter '\n' to other tokens?
            print('prefix_text:', prefix_text)
            prefix_text = text_convert(prefix_text, self.text_norm)
            prefix_text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(prefix_text)
            st_pos = len(prefix_text)
            print(st_pos)
            text = text_convert(text, self.text_norm)
            text = prefix_text + self.tokenizer.encode(self.sep + text)
            print(text)
            labels = text[1:] + [self.tokenizer.eot]
            print(len(labels))
        else:
            text = text_convert(text, self.text_norm)
            if self.fix_tokenizer:
                text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(' ' + text)
            else:
                text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
            labels = text[1:] + [self.tokenizer.eot]

        if self.reference_dict:
            reference = self.reference_dict[audio_id]
            text = [self.tokenizer.sot_prev] + self.tokenizer.encode(' ' + reference) + text
            labels = len([self.tokenizer.sot_prev] + self.tokenizer.encode(' ' + reference)) * [-100] + labels

        if self.init_prompt:
            print(text)
            text = [self.tokenizer.sot_prev] + self.tokenizer.encode(' ' + self.init_prompt) + text
            print(text)
            print(labels)
            labels = len([self.tokenizer.sot_prev] + self.tokenizer.encode(' ' + self.init_prompt)) * [-100] + labels
            print(labels)

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            'st_pos': st_pos,
        }


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50256) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id, TODO: multilingual models?

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


def plot_attention(weights, audio_id, name, xlim=None):
    # fig, ax = plt.subplots(figsize=(15, 3))
    fig, ax = plt.subplots()
    ax.imshow(weights.cpu(), cmap="viridis")
    ax.invert_yaxis()
    #ax.stem(weights)
    ax.set_xlabel("Encoder outputs")
    ax.set_ylabel("Step")
    if xlim:
        plt.xlim(0, xlim)
    plt.plot()
    # ax.set_yticks(range(12))
    plt.savefig(name)
    #plt.savefig(f'pics/{audio_id}_{name}.png')


def calc_attention(args, wtokenizer, nlayers=12):
    # Test dataset & dataloader
    eval_datalist = load_audio_file_list(args.eval_list_file)

    woptions = whisper.DecodingOptions(language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'))
    if args.n_decoder_prompts:
        wmodel = whisper.load_soft_model(args.model, device=args.device, download_root=args.model_dir, n_decoder_prompts=args.n_decoder_prompts, v2=args.v2, dropout=args.dropout)
        init_embedding(wmodel, wtokenizer, args.n_decoder_prompts, args.device, args.initial_prompt_scheme, args.train_list_file)
    else:
        wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout)
    dataset = SpeechDataset(eval_datalist, wtokenizer, args.sample_rate, args, args.text_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=WhisperDataCollatorWhithPadding())
    wmodel.eval()

    if args.checkpoint:
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}', map_location=torch.device('cpu'))
        state_dict = state_dict['state_dict']
        state_dict_new = {}
        for k in state_dict:
            state_dict_new[k[6:]] = state_dict[k]
        if args.n_decoder_prompts:
            #state_dict['decoder.learned_embedding'] = state_dict['model.decoder.learned_embedding']
            wmodel.load_state_dict(state_dict_new, strict=False) # only load soft prompt embeddings
        else:
            wmodel.load_state_dict(state_dict_new)

    print('========')
    for audio_id, audio_path, text in eval_datalist:
        result = wmodel.transcribe(audio_path, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'), suppress_blank=args.suppress_blank, condition_on_previous_text=args.condition_on_previous_text, args=args)
        text = text_convert(text, args.text_norm)
        # print(result)
        segments = []
        attn_weights = []
        for segment in result["segments"]:
            segments.append(segment['text'].strip())
            attn_weight = segment['attn_weights']
            attn_layer = []
            for ii in range(nlayers):
                attn_layer.append(torch.mean(torch.stack(attn_weight[str(ii)], dim=0), dim=1))
            wlayer = torch.mean(torch.stack(attn_layer), dim=0)
            print(segment['start'], segment['end'])
            wlayer = wlayer[:, int(segment['start'] * 50):int(segment['end'] * 50)]
            attn_weights.append(wlayer)
        final_weight = torch.cat(attn_weights, dim=1)
        print(final_weight.shape)
        final_weight = torch.sum(final_weight, dim=0).unsqueeze(0).expand(20, -1)
        #print(final_weight.shape)
        # print(final_weight[0])
        '''
        if args.checkpoint == False:
            name = 'baseline'
        elif args.n_decoder_prompts:
            name = f'sp_{args.n_decoder_prompts}'
        else:
            name = 'finetune'
        '''
        check_output_dir(f'{args.outdir}/pics')
        plot_attention(final_weight.cpu(), audio_id, f'{args.outdir}/pics/{audio_id}.png')
        print(' '.join(segments))
        print(text)

    # print('========')
    # for audio_id, audio_path, text in eval_datalist:
    #     result = wmodel.transcribe(audio_path, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'), suppress_blank=args.suppress_blank)
    #     text = text_convert(text, args.text_norm)
    #     print(result)
    #     segments = []
    #     attn_weights = []
    #     for segment in result["segments"]:
    #         segments.append(segment['text'].strip())
    #         attn_weight = segment['attn_weights']
    #         attn_layer = []
    #         for ii in range(nlayers):
    #             attn_layer.append(torch.sum(torch.stack(attn_weight[str(ii)]), dim=0))
    #         wlayer = torch.cat(attn_layer, dim=0)
    #         print(wlayer.shape)
    #         # wlayer = wlayer[:, int(segment['start'] * 50):int(segment['end'] * 50)]
    #         print(wlayer.shape)
    #         attn_weights.append(wlayer)
    #     final_weight = torch.cat(attn_weights, dim=1)
    #     print(final_weight.shape)
    #     print(final_weight[0])
    #     plot_attention(final_weight)
    #     print(' '.join(segments))
    #     print(text)


def rescoring(args, wtokenizer):
    # TODO: change out_fname, outdir???
    out_dir = f"{os.path.split(args.eval_list_file)[0].replace('data', 'exp')}/{args.model}"
    check_output_dir(out_dir)
    out_fname = f"{out_dir}/{args.eval_list_file.split('/')[-1]}"
    if args.initial_prompt_scheme:
        out_fname += '_' + args.initial_prompt_scheme.split('/')[-1]
    if args.few_shot_path:
        out_fname += '_' + args.few_shot_path.split('/')[-1]
    if args.ilm:
        if args.ilm == 'gaussian':
            out_fname += '_ilm_gaussian_' + str(args.std) + '_' + str(args.seed)
        elif args.ilm == 'zero':
            out_fname += '_ilm_zeros'
        elif args.ilm == 'nocross':
            out_fname += '_ilm_nocross'
        elif args.ilm == 'avgh':
            out_fname += '_ilm_avgh'
        else:
            assert 0
    if args.eval_dataset_name:
        out_fname += '__' + args.eval_dataset_name

    if args.ilm and args.ilm != 'avgh':
        eval_datalist = load_audio_file_list(args.eval_list_file, ilm=True)
    else:
        eval_datalist = load_audio_file_list(args.eval_list_file)
    woptions = whisper.DecodingOptions(language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'))
    if args.ilm == 'nocross' or args.ilm == 'avgh':
        wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=args.ilm, lora=args.lora)
    else:
        wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=False, lora=args.lora)

    if args.checkpoint:
        if args.n_decoder_prompts:
            whisper_model = SoftWhisperModelModule(args)
        else:
            whisper_model = WhisperModelModule(args)
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}')
        print('load state dict:', f'{args.outdir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings
        wmodel = whisper_model.model
        dirA, dirB = args.outdir.split('/')[-2], args.outdir.split('/')[-1]
        out_fname += '__' + dirA + '__' + dirB

    print(args.device, out_fname)
    if args.ilm == 'gaussian':
        audio = torch.from_numpy(numpy.float32(numpy.random.normal(0, args.std, size=int(args.avg_len * 16000))))
    elif args.ilm == 'zero':
        audio = torch.zeros(int(args.avg_len * 16000))
    else:
        audio = None

    if args.initial_prompt_scheme:
        with open(args.initial_prompt_scheme) as fin:
            init_prompt = fin.readline().strip()
        print(init_prompt)
    else:
        init_prompt = None

    if args.few_shot_path:
        with open(args.few_shot_path) as fin:
            few_shot_lst = load_audio_file_list(args.few_shot_path)
    else:
        few_shot_lst = None

    dataset = SpeechDataset(eval_datalist, wtokenizer, args.sample_rate, args, args.text_norm, args.eval_ref_file, audio=audio, init_prompt=init_prompt, few_shot_lst=few_shot_lst, fix_tokenizer=args.fix_tokenizer, n_mels=wmodel.dims.n_mels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=WhisperDataCollatorWhithPadding())
    wmodel.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    with open(out_fname, 'w') as fout:
        idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].long().to(args.device)
            dec_input_ids = batch["dec_input_ids"].long().to(args.device)

            with torch.no_grad():
                audio_features = wmodel.encoder(input_ids)
                out = wmodel.decoder(dec_input_ids, audio_features)
                if args.n_decoder_prompts:
                    out = out[:, args.n_decoder_prompts:, :]
                loss = loss_fn(out.reshape(-1, out.size(-1)), labels.view(-1)).view(dec_input_ids.size(0), -1)
                loss = loss.sum(dim=-1)

            for l, label in zip(loss, labels):
                audio_id, audio_path, text = eval_datalist[idx]
                fout.write(f'{audio_id} {audio_path} {float(l)} {text}\n')
                # label = [t for t in label if t > 0 and t < wtokenizer.eot]
                # print(text, wtokenizer.decode(label))
                idx = idx + 1


def rescoring_plt(args, wtokenizer, n_layers=32):
    # TODO: change out_fname, outdir???
    out_dir = f"{os.path.split(args.eval_list_file)[0].replace('data', 'exp')}/{args.model}"
    check_output_dir(out_dir)
    out_fname = f"{out_dir}/{args.eval_list_file.split('/')[-1]}"
    if args.ilm:
        if args.ilm == 'gaussian':
            out_fname += '_ilm_gaussian_' + str(args.std) + '_' + str(args.seed)
        elif args.ilm == 'zero':
            out_fname += '_ilm_zeros'
        elif args.ilm == 'nocross':
            out_fname += '_ilm_nocross'
        elif args.ilm == 'avgh':
            out_fname += '_ilm_avgh'
        else:
            assert 0
    print(args.device, out_fname)
    if args.eval_dataset_name:
        out_fname += '__' + args.eval_dataset_name
    if args.ilm and args.ilm != 'avgh':
        eval_datalist = load_audio_file_list(args.eval_list_file, ilm=True)
    else:
        eval_datalist = load_audio_file_list(args.eval_list_file)
    woptions = whisper.DecodingOptions(language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'))
    if args.ilm == 'nocross' or args.ilm == 'avgh':
        wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=args.ilm, lora=args.lora)
    else:
        wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=False, lora=args.lora)
    # kv_cache, hooks = wmodel.install_kv_cache_hooks()

    if args.ilm == 'gaussian':
        audio = torch.from_numpy(numpy.float32(numpy.random.normal(0, args.std, size=int(args.avg_len * 16000))))
    elif args.ilm == 'zero':
        audio = torch.zeros(int(args.avg_len * 16000))
    else:
        audio = None

    dataset = SpeechDataset(eval_datalist, wtokenizer, args.sample_rate, args, args.text_norm, args.eval_ref_file, audio=audio)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=WhisperDataCollatorWhithPadding())
    wmodel.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    with open(out_fname, 'w') as fout:
        idx = 0
        for batch in loader:
            kv_cache = dict()
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].long().to(args.device)
            dec_input_ids = batch["dec_input_ids"].long().to(args.device)
            print(labels)

            with torch.no_grad():
                audio_features = wmodel.encoder(input_ids)
                out = wmodel.decoder(dec_input_ids, audio_features, kv_cache=kv_cache)
                loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1)).view(dec_input_ids.size(0), -1)
                loss = loss.sum(dim=-1)

            attn_weights = []
            for ii in range(n_layers):
                weight = kv_cache[str(ii)].squeeze()
                weight = weight[:, int(args.std), :int(args.avg_len*50)]
                attn_weights.append(weight)
            final_weight = torch.cat(attn_weights, dim=0)
            # print(final_weight)
            print(final_weight.shape)
            # final_weight = final_weight.mean(dim=0)
            # final_weight = final_weight.unsqueeze(0).expand(20, -1)
            check_output_dir(f'{args.outdir}/pics')
            plot_attention(final_weight.cpu(), time.asctime(), f'{args.outdir}/pics/{time.asctime()}.png', int(args.avg_len*50))
            # final_weight = torch.sum(final_weight, dim=0).unsqueeze(0).expand(20, -1)
            # plot_attention(final_weight.cpu(), time.asctime(), f'{args.outdir}/pics/{time.asctime()}.png')

            for l, label in zip(loss, labels):
                audio_id, audio_path, text = eval_datalist[idx]
                fout.write(f'{audio_id} {audio_path} {float(l)} {text}\n')
                # label = [t for t in label if t > 0 and t < wtokenizer.eot]
                # print(text, wtokenizer.decode(label))
                idx = idx + 1


class WhisperModelModule(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, fp16=(args.device!='cpu'), suppress_blank=args.suppress_blank)
        self.model = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=args.ilm, lora=args.lora, attention_window=args.attention_window, n_ctx=args.chunk_size * TOKENS_PER_SECOND, n_text_ctx=args.n_text_ctx)
        self.tokenizer = whisper.tokenizer.get_tokenizer(not args.model.endswith('.en'), language=args.language, task='transcribe', num_languages=self.model.num_languages)

        if args.lora:
            loralib.mark_only_lora_as_trainable(self.model)
            # for n, p in self.model.encoder.named_parameters():
            #     print(n, p.requires_grad)

        if args.frozen == 'encoder':
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        elif args.frozen == 'decoder':
            for p in self.model.decoder.parameters():
                p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.args = args
        self.__train_dataset = load_audio_file_list(args.train_list_file)

        if args.language == 'en' or args.task == 'translate':
            self.std = EnglishTextNormalizer()
        else:
            self.std = BasicTextNormalizer()
        self.save_hyperparameters()
        self.recog_sentences = defaultdict(list)

        self.best_loss = math.inf
        self.best_ckpt = ''
        self.best_file = os.path.join(args.outdir, 'best_ckpt')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        wandb.log({'train/loss': loss, 'train/learning_rate': self.optimizer.param_groups[0]['lr']})
        # print(loss)
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            ind = 0
            for i, t in enumerate(l):
                if t == wtokenizer.no_timestamps:
                    ind = i
                    break
            o = o[ind:]
            o = [t for t in o if t < wtokenizer.eot]
            l = [t for t in l if t < wtokenizer.eot]
            o = o[:len(l)]
            hyp = self.tokenizer.decode(o) #, skip_special_tokens=True)
            ref = self.tokenizer.decode(l) #, skip_special_tokens=True)
            self.recog_sentences[dataloader_idx].append([hyp.strip(), ref.strip()])

    def on_validation_epoch_end(self):
        wers = []
        for dataloader_idx in self.recog_sentences:
            errors, refs = 0, 0
            with open(os.path.join(self.args.outdir, 'validation', f'step{self.global_step}_{dataloader_idx}'), 'w') as fout:
                for hyp, ref in self.recog_sentences[dataloader_idx]:
                    fout.write(f'hyp: {hyp}\n')
                    fout.write(f'ref: {ref}\n')
                    hyp_tn = self.std(hyp)
                    ref_tn = self.std(ref)
                    errors += editdistance.eval(hyp_tn.split(), ref_tn.split())
                    refs += len(ref_tn.split())
            wers.append([errors, refs])
            print('wer:', errors / refs)
            wandb.log({f'val{dataloader_idx}/wer': errors/refs}, commit=False)
            self.recog_sentences[dataloader_idx].clear()

        avg_wer = sum([x[0] for x in wers]) / sum(x[1] for x in wers)
        last_ckpt = f'epoch={self.current_epoch}-step={self.global_step}.ckpt'
        if avg_wer < self.best_loss:
            self.best_loss = avg_wer
            self.best_ckpt = last_ckpt
            with open(self.best_file, 'w') as f:
                f.write(f'{last_ckpt}\t{avg_wer}')

        if self.global_step and not self.args.n_decoder_prompts:
            # Only save the best checkpoint or last checkpoint
            for fname in os.listdir(os.path.join(args.outdir, 'checkpoint')):
                print(fname, last_ckpt, self.best_ckpt)
                if fname.startswith('epoch='):
                    current = int(fname.split('=')[-1].split('.')[0])
                    last = int(last_ckpt.split('=')[-1].split('.')[0])
                    best = int(self.best_ckpt.split('=')[-1].split('.')[0])
                    if abs(current-last) >= 100 and abs(current-best) >= 100:
                        os.remove(os.path.join(args.outdir, 'checkpoint', fname))
                # if fname.startswith('epoch=') and fname != last_ckpt and fname != self.best_ckpt:
                #     os.remove(os.path.join(args.outdir, 'checkpoint', fname))
                    # print('remove', fname)

    def configure_optimizers(self):
        """Initialize optimizer and scheduler"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.args.learning_rate, 
                          eps=self.args.adam_epsilon)
        self.optimizer = optimizer

        if self.args.constant_lr:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps, 
                num_training_steps=self.t_total
            )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        """Initialization (Load Dataset)"""
        if stage == 'fit' or stage is None:
            if self.args.num_steps > 0:
                self.t_total = self.args.num_steps
            else:
                self.t_total = (
                    (len(self.__train_dataset) // (self.args.train_batch_size))
                    // self.args.gradient_accumulation_steps
                    * float(self.args.num_train_epochs)
                )

    def train_dataloader(self):
        """Create training dataloader"""
        dataset = SpeechDataset(self.__train_dataset, self.tokenizer, self.args.sample_rate, args, self.args.text_norm, args.train_ref_file, fix_tokenizer=args.fix_tokenizer)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.args.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.args.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


class WhisperILMModelModule(WhisperModelModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = whisper.load_soft_model(args.model, device=args.device, download_root=args.model_dir, n_decoder_prompts=args.n_decoder_prompts, v2=args.v2, dropout=args.dropout)


class SoftWhisperModelModule(WhisperModelModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = whisper.load_soft_model(args.model, device=args.device, download_root=args.model_dir, n_decoder_prompts=args.n_decoder_prompts, v2=args.v2, dropout=args.dropout)
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = False
        self.model.decoder.learned_embedding.requires_grad = True
        init_embedding(self.model, self.tokenizer, args.n_decoder_prompts, args.device, args.initial_prompt_scheme, args.train_list_file)

    # Only save learned_embeddings
    def on_save_checkpoint(self, checkpoint) -> None:
        del checkpoint['state_dict']
        checkpoint['state_dict'] = collections.OrderedDict()
        checkpoint['state_dict']['model.decoder.learned_embedding'] = self.model.state_dict()['decoder.learned_embedding']

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # TODO: remove this line for encoder-side prompts
        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        out = out[:, args.n_decoder_prompts:, :]
        loss = self.loss_fn(out.reshape(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        wandb.log({'train/loss': loss, 'train/learning_rate': self.optimizer.param_groups[0]['lr']})
        # print(loss)
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)
        out = out[:, args.n_decoder_prompts:, :]

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot
        # loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            ind = 0
            for i, t in enumerate(l):
                if t == wtokenizer.no_timestamps:
                    ind = i
                    break
            o = o[ind:]
            o = [t for t in o if t < wtokenizer.eot]
            l = [t for t in l if t < wtokenizer.eot]
            o = o[:len(l)]
            hyp = self.tokenizer.decode(o) #, skip_special_tokens=True)
            ref = self.tokenizer.decode(l) #, skip_special_tokens=True)
            self.recog_sentences[dataloader_idx].append([hyp, ref])


def train(args, wtokenizer):
    wandb.init(
        project='soft_prompts',
        entity='rao-mengjie',
        name='|'.join(args.outdir.split('/')[-2:]),
        config=args,
        # mode='disabled'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.outdir}/checkpoint",
        every_n_train_steps=args.save_step_frequency,
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback]
    check_output_dir(f'{args.outdir}/validation')

    if args.n_decoder_prompts:
        print(args.n_decoder_prompts)
        model = SoftWhisperModelModule(args)
    else:
        model = WhisperModelModule(args)

    if args.init_ckpt:
        best_ckpt = open(os.path.join(args.init_ckpt, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.init_ckpt}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict) # only load soft prompt embeddings
        print('... load successfully ...')

    if args.num_steps > 0:
        args.num_train_epochs = None

    trainer = Trainer(
        precision=16,
        accelerator=args.device,
        max_epochs=args.num_train_epochs,
        max_steps=args.num_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=callback_list,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        default_root_dir=args.outdir
    )

    val_dataset = SpeechDataset(load_audio_file_list(args.eval_list_file), wtokenizer, args.sample_rate, args, args.text_norm, args.eval_ref_file, fix_tokenizer=args.fix_tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_worker,
                        collate_fn=WhisperDataCollatorWhithPadding()
                        )

    test_dataset = SpeechDataset(load_audio_file_list(args.test_list_file), wtokenizer, args.sample_rate, args, args.text_norm, args.test_ref_file, fix_tokenizer=args.fix_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_worker,
                        collate_fn=WhisperDataCollatorWhithPadding()
                        )

    # calculate WER before starting training
    # trainer.validate(model, dataloaders=[val_dataloader, test_dataloader])
    trainer.fit(model, val_dataloaders=[val_dataloader, test_dataloader],
                ckpt_path='last' if args.checkpoint else '')
    wandb.finish()


def evaluate(args, wtokenizer, list_path, dataset_name, ref_file=None):
    if args.n_decoder_prompts:
        whisper_model = SoftWhisperModelModule(args)
    else:
        whisper_model = WhisperModelModule(args)
    if args.checkpoint:
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        # print(state_dict.keys())
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings
        whisper_model = whisper_model.to(args.device)

    if args.language == 'en' or args.task == 'translate':
        std = EnglishTextNormalizer()
    elif args.language == 'zh':
        std = BasicTextNormalizer(split_letters=True)
    else:
        std = BasicTextNormalizer()
    eval_datalist = load_audio_file_list(list_path)
    check_output_dir(f'{args.outdir}/{args.task}')

    reference_dict = {}
    if ref_file:
        for line in open(ref_file):
            line = line.strip()
            if '-hyp' in line:
                audio_id = line.split()[0][:-5]
                if len(line.split()) == 1:
                    text = ''
                else:
                    text = line.split(None, 1)[1]
            elif '-ref' in line:
                pass
            else:
                audio_id = line.split()[0]
                text = line.split(None, 2)[-1]
            reference_dict[audio_id] = text.lower()

    whisper_model.eval()
    out_fname = f'{args.outdir}/{args.task}/{args.checkpoint}_{dataset_name}_beam{args.beam_size}_stamp{not args.notimestamp}'
    if args.length_penalty:
        out_fname += f'_lp_{args.length_penalty}'
    if not args.suppress_blank:
        out_fname += '_no_sup_blank'
    out_no_norm_fname = out_fname + '_nonorm'
    with open(out_fname, 'w') as fout, open(out_no_norm_fname, 'w') as forigin:
        result_list = []
        for audio_id, audio_path, text in eval_datalist:
            if reference_dict:
                reference = reference_dict[audio_id]
            else:
                reference = None
            if text.startswith('st:'):
                st_time, text = text.split(None, 1)
                ed_time, text = text.split(None, 1)
                st_time = int(float(st_time[3:]) * SAMPLE_RATE)
                ed_time = int(float(ed_time[3:]) * SAMPLE_RATE)
            else:
                st_time, ed_time = -1, -1
            with torch.no_grad():
                result = whisper_model.model.transcribe(audio_path, task=args.task, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, length_penalty=args.length_penalty, fp16=(args.device!='cpu' and not args.lora), condition_on_previous_text=args.condition_on_previous_text, initial_prompt=reference, args=args, suppress_tokens=args.suppress_tokens, st_time=st_time, ed_time=ed_time)
                segments = []
                for segment in result['segments']:
                    segments.append(segment['text'].strip())
                    print(segment['start'], segment['end'], segment['text'])
                result_list.append([audio_id, ' '.join(segments), text_convert(text, norm=args.text_norm)])

        o_errors, o_refs = 0, 0
        errors, refs = 0, 0
        for audio_id, hyp, ref in result_list:
            forigin.write(f'{audio_id}-hyp: {hyp}\n')
            forigin.write(f'{audio_id}-ref: {ref}\n')
            o_errors += editdistance.eval(hyp.split(), ref.split())
            o_refs += len(ref.split())

            # Added for Linguaskill test set
            ref = ref.replace('%hesitation%', '')
            ref = re.sub(r'(\w+-)(\s|$)', r'', ref)
            hyp = hyp.replace('%hesitation%', '')
            hyp = re.sub(r'(\w+-)(\s|$)', r'', hyp)
            hyp_tn, ref_tn = std(hyp), std(ref)
            fout.write(f'{audio_id}-hyp: {hyp_tn}\n')
            fout.write(f'{audio_id}-ref: {ref_tn}\n')
            errors += editdistance.eval(hyp_tn.split(), ref_tn.split())
            refs += len(ref_tn.split())

        o_wers = o_errors / o_refs
        print(o_errors, o_refs, o_wers)
        forigin.write(f'=== errors: {o_errors}, refs: {o_refs}, wers: {o_wers} ===\n')
        wers = errors / refs
        print(errors, refs, wers)
        fout.write(f'=== errors: {errors}, refs: {refs}, wers: {wers} ===\n')

        if args.task == 'translate':
            bleu = heval.load('bleu')
            predictions = [hyp for _, hyp, _ in result_list]
            references = [[ref] for _, _, ref in result_list]
            eval_results = bleu.compute(predictions=predictions, references=references)
            print(eval_results["bleu"])
            fout.write(f'=== bleu: {eval_results["bleu"]} ===\n')

            predictions = [std(hyp) for _, hyp, _ in result_list]
            references = [[std(ref)] for _, _, ref in result_list]
            eval_results = bleu.compute(predictions=predictions, references=references)
            print(eval_results)
            print(eval_results["bleu"])
            fout.write(f'=== bleu: {eval_results["bleu"]} ===\n')


def gen_cem_data(args, wtokenizer, list_path, dataset_name, ref_file=None):
    if args.n_decoder_prompts:
        whisper_model = SoftWhisperModelModule(args)
    else:
        whisper_model = WhisperModelModule(args)
    if args.checkpoint:
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings
        whisper_model = whisper_model.to(args.device)

    eval_datalist = load_audio_file_list(list_path)
    check_output_dir(f'{args.outdir}/{args.task}')
    whisper_model.eval()
    out_fname = f'cem_data/{dataset_name}_beam{args.beam_size}.csv'
    print(f'Outdir: {out_fname}')

    reference_dict = {}
    if ref_file:
        for line in open(ref_file):
            line = line.strip()
            if '-hyp' in line:
                audio_id = line.split()[0][:-5]
                if len(line.split()) == 1:
                    text = ''
                else:
                    text = line.split(None, 1)[1]
            elif '-ref' in line:
                pass
            else:
                audio_id = line.split()[0]
                text = line.split(None, 2)[-1]
            reference_dict[audio_id] = text.lower()
    
    with open(out_fname, 'w', newline='') as fout:
        csv_writer = csv.writer(fout)
        for audio_id, audio_path, text in tqdm(eval_datalist):
            if reference_dict:
                reference = reference_dict[audio_id]
            else:
                reference = None
            if text.startswith('st:'):
                st_time, text = text.split(None, 1)
                ed_time, text = text.split(None, 1)
                st_time = int(float(st_time[3:]) * SAMPLE_RATE)
                ed_time = int(float(ed_time[3:]) * SAMPLE_RATE)
            else:
                st_time, ed_time = -1, -1
            with torch.no_grad():
                result = whisper_model.model.transcribe(audio_path, for_cem=True, task=args.task, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, length_penalty=args.length_penalty, fp16=(args.device!='cpu' and not args.lora), condition_on_previous_text=args.condition_on_previous_text, initial_prompt=reference, args=args, suppress_tokens=args.suppress_tokens, st_time=st_time, ed_time=ed_time)
                tokens = []
                attn_features = []
                dec_features = []
                sm_probs = []
                for segment in result['segments']:
                    tokens.extend(segment['tokens'])
                    attn_features.extend(torch.unbind(segment['attn_states']))
                    dec_features.extend(torch.unbind(segment['dec_states']))
                    sm_probs.extend([np.exp(prob) for prob in segment['log_token_probs']])
                
            # obtain correctness lavels for tokens
            ref_txt = text_convert(text, norm=args.text_norm)
            ref = wtokenizer.encode(ref_txt)
            token_truth_labels, incorrect = token_align(tokens, ref, wtokenizer)
            
            for token, attn_state, dec_state, sm_prob, label in zip(tokens, attn_features, dec_features, sm_probs, 
                                                                    token_truth_labels):
                emb = whisper_model.model.decoder.token_embedding(torch.tensor([token]).to('cuda')).to('cpu')
                emb = torch.squeeze(emb)
                data = torch.cat((torch.tensor([token]), attn_state, dec_state, emb, 
                                    torch.tensor([sm_prob]), torch.tensor([label]))).tolist()
                csv_writer.writerow(data)

  

def evaluate_segment(args, wtokenizer, list_path, dataset_name):
    if args.n_decoder_prompts:
        whisper_model = SoftWhisperModelModule(args)
    else:
        whisper_model = WhisperModelModule(args)
    if args.checkpoint:
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings

    std = EnglishTextNormalizer()
    eval_datalist = load_audio_file_list(list_path)
    check_output_dir(f'{args.outdir}/segment_transcribe')

    whisper_model.eval()
    out_fname = f'{args.outdir}/segment_transcribe/{args.checkpoint}_{dataset_name}_beam{args.beam_size}_stamp{not args.notimestamp}'
    if args.length_penalty:
        out_fname += f'_lp_{args.length_penalty}'
    if not args.suppress_blank:
        out_fname += '_no_sup_blank'
    out_no_norm_fname = out_fname + '_nonorm'
    with open(out_fname, 'w') as fout, open(out_no_norm_fname, 'w') as forigin:
        result_list = defaultdict(list)
        for audio_id, audio_path, text in eval_datalist:
            with torch.no_grad():
                result = whisper_model.model.transcribe(audio_path, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, length_penalty=args.length_penalty, fp16=(args.device!='cpu'), condition_on_previous_text=args.condition_on_previous_text, args=args)
                segments = []
                for ii, segment in enumerate(result['segments']):
                    segments.append(segment['text'].strip())
                    print(segment['text'].strip())
                    result_list[audio_id].append([ii, segment['text'].strip(), text_convert(text, norm=args.text_norm)])

        o_errors, o_refs = 0, 0
        errors, refs = 0, 0
        for audio_id in result_list:
            hyp_list = []
            hyp_tn_list = []
            for ii, hyp, ref in result_list[audio_id]:
                forigin.write(f'{audio_id}-{ii}-hyp: {hyp}\n')
                # Added for Linguaskill test set
                ref = ref.replace('%hesitation%', '')
                ref = re.sub(r'(\w+-)(\s|$)', r'', ref)
                hyp = hyp.replace('%hesitation%', '')
                hyp = re.sub(r'(\w+-)(\s|$)', r'', hyp)
                hyp_tn, ref_tn = std(hyp), std(ref)
                fout.write(f'{audio_id}-{ii}-hyp: {hyp_tn}\n')
                hyp_list.append(hyp.strip())
                hyp_tn_list.append(hyp_tn.strip())

            hyp = ' '.join(hyp_list)
            hyp_tn = ' '.join(hyp_tn_list)
            forigin.write(f'{audio_id}-ref: {ref}\n')
            fout.write(f'{audio_id}-ref: {ref_tn}\n')
            o_errors += editdistance.eval(hyp.split(), ref.split())
            o_refs += len(ref.split())
            errors += editdistance.eval(hyp_tn.split(), ref_tn.split())
            refs += len(ref_tn.split())

        o_wers = o_errors / o_refs
        print(o_errors, o_refs, o_wers)
        forigin.write(f'=== errors: {o_errors}, refs: {o_refs}, wers: {o_wers} ===\n')
        wers = errors / refs
        print(errors, refs, wers)
        fout.write(f'=== errors: {errors}, refs: {refs}, wers: {wers} ===\n')


def analyse_text_content(wmodel, wtokenizer, list_path):
    if args.n_decoder_prompts:
        whisper_model = SoftWhisperModelModule(args)
    else:
        whisper_model = WhisperModelModule(args)
    if args.checkpoint:
        best_ckpt = open(os.path.join(args.outdir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.outdir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings

    std = EnglishTextNormalizer()
    std_num = EnglishNumberNormalizer()
    eval_datalist = load_audio_file_list(list_path)
    check_output_dir(f'{args.outdir}/segment_transcribe')

    whisper_model.eval()
    diff_count = 0
    seg_count = 0
    for audio_id, audio_path, text in eval_datalist:
        result_list = defaultdict(list)
        with torch.no_grad():
            result = whisper_model.model.transcribe(audio_path, language=args.language, without_timestamps=args.notimestamp, beam_size=args.beam_size, length_penalty=args.length_penalty, fp16=(args.device!='cpu'), condition_on_previous_text=args.condition_on_previous_text, args=args)
            for ii, segment in enumerate(result['segments']):
                result_list[audio_id].append([ii, segment['text'].strip(), text_convert(text, norm=args.text_norm)])
            
            for audio_id in result_list:
                hyp_list = []
                for ii, hyp, ref in result_list[audio_id]:
                    hyp = std_num(hyp)
                    ref = std_num(ref)
                    if hyp != std(hyp):
                        diff_count += 1
                    seg_count +=1
                    print(hyp)
                    print(f'Current non-norm rate {diff_count/seg_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe wav files in a wav list')
    parser.add_argument('--model', type=str, default='small.en', help='name of the pretrained model')
    parser.add_argument('--outdir', type=str, default='', help='output directory to save the transcribed data in jsonl format')
    parser.add_argument('--checkpoint', type=str2bool, default=True, help='Saved checkpoint path')
    parser.add_argument('--init_ckpt', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='/home/mifs/th624/.cache/whisper', help='path to load model')
    parser.add_argument('--beam_size', type=int, default=5, help='output nbest with logprob for each token')
    parser.add_argument('--seed', type=int, default=1031, help='random seed')
    parser.add_argument('--notimestamp', type=str2bool, default=True, help='True: without_timestamps=True')
    parser.add_argument('--suppress_blank', type=str2bool, default=True)
    parser.add_argument('--condition_on_previous_text', type=str2bool, default=True)
    parser.add_argument('--v2', type=str2bool, default=False)
    parser.add_argument('--ilm', type=str, default=None)
    parser.add_argument('--text_norm', type=str, default='lower')
    parser.add_argument('--initial_prompt_scheme', type=str, default=None)
    parser.add_argument('--few_shot_path', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--constant_lr', type=str2bool, default=False, help='True: use constant learning rate')
    parser.add_argument('--length_penalty', type=float, default=None)
    parser.add_argument('--train_list_file', type=str, default='/scratches/dialfs/mvse/mq227/whisper_mrx/linguaskill_data/train_combine/audio_ref_pair_list_20h')
    parser.add_argument('--analyse_list_file', type=str, default='/scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/train_flt.tsv')
    parser.add_argument('--eval_list_file', type=str, default='/scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_single_flt.tsv')
    parser.add_argument('--cem_list_file', type=str, default='/scratches/dialfs/alta/th624/exp-th624/data/Linguaskill/flt/test_flt.tsv')
    parser.add_argument('--test_list_file', type=str, default='')
    parser.add_argument('--train_ref_file', type=str, default='')
    parser.add_argument('--eval_ref_file', type=str, default='')
    parser.add_argument('--test_ref_file', type=str, default='')
    parser.add_argument('--enable_specaug', type=str2bool, default=False)
    parser.add_argument('--fix_tokenizer', type=str2bool, default=False)
    parser.add_argument('--lora', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=30)
    parser.add_argument('--n_text_ctx', type=int, default=448)
    parser.add_argument('--max_time_warp', type=int, default=80)
    parser.add_argument('--max_freq_width', type=int, default=27)
    parser.add_argument('--n_freq_mask', type=int, default=2)
    parser.add_argument('--max_time_width', type=int, default=100)
    parser.add_argument('--n_time_mask', type=int, default=2)
    parser.add_argument('--frozen', type=str, default='', help='frozen some parameters in the training')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--val_check_interval', type=int, default=2500)
    parser.add_argument('--save_step_frequency', type=int, default=2500)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=-1)
    parser.add_argument('--suppress_tokens', type=str, default="-1")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--n_decoder_prompts', type=int, default=0)
    parser.add_argument('--attention_window', type=int, default=-1)
    parser.add_argument('--avg_len', type=float, default=0)
    parser.add_argument('--std', type=float, default=0)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--eval_dataset_name', type=str, default='dev_clean')
    parser.add_argument('--test_dataset_name', type=str, default='dev_other')
    parser.add_argument('--cem_dataset_name', type=str, default='test')
    parser.add_argument('--stage', type=str, default='debug')
    parser.add_argument('--task', type=str, default='transcribe')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.beam_size:
        args.beam_size = None
    if args.stage == 'rescoring' or args.stage == 'rescoring_plt':
        pass
    elif args.stage == 'evaluate' and not args.checkpoint:
        args.outdir = f'exp/baseline/{args.model}/{args.outdir}'
    else:
        args.outdir = f'exp/{args.model}/{args.outdir}'
    # print(args)

    if args.outdir:
        check_output_dir(args.outdir)
    seed_everything(args.seed, workers=True)

    wmodel = whisper.load_model(args.model, device=args.device, download_root=args.model_dir, dropout=args.dropout, ilm=False, lora=args.lora)
    wtokenizer = whisper.tokenizer.get_tokenizer(not args.model.endswith('.en'), language=args.language, task='transcribe', num_languages=wmodel.num_languages)

    if args.stage == 'debug':
        debug(args, wtokenizer)
    elif args.stage == 'rescoring':
        rescoring(args, wtokenizer)
    elif args.stage == 'rescoring_plt':
        rescoring_plt(args, wtokenizer)
    elif args.stage == 'attention':
        calc_attention(args, wtokenizer)
    elif args.stage == 'train':
        train(args, wtokenizer)
    elif args.stage == 'evaluate':
        if args.eval_list_file:
            evaluate(args, wtokenizer, args.eval_list_file, args.eval_dataset_name, args.eval_ref_file)
        if args.test_list_file:
            evaluate(args, wtokenizer, args.test_list_file, args.test_dataset_name, args.test_ref_file)
    elif args.stage == 'evaluate_segment':
        if args.eval_list_file:
            evaluate_segment(args, wtokenizer, args.eval_list_file, args.eval_dataset_name, args.eval_ref_file)
        if args.test_list_file:
            evaluate_segment(args, wtokenizer, args.test_list_file, args.test_dataset_name, args.test_ref_file)
    elif args.stage == 'gen_cem_data':
        if args.eval_list_file:
            gen_cem_data(args, wtokenizer, args.cem_list_file, args.cem_dataset_name)
    elif args.stage == 'analyse_text':
            analyse_text_content(args, wtokenizer, args.analyse_list_file)


    else:
        assert 0
