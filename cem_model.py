import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import torch.nn.functional as F
import gc


class SimpleCEM(nn.Module):
    def __init__(self, n: int, feature_mode: str = 'no_dec', return_logits: bool = False,
                 norm: bool = False, whisper_model=None, deep=0):
        super().__init__()
        if feature_mode == 'emb_only' or feature_mode == 'attn_only' or feature_mode == 'dec_only':
            inp_size = 769
        elif feature_mode == 'no_dec':
            inp_size = 1537
        elif feature_mode == 'no_prob':
            inp_size = 2304
        elif feature_mode == 'all':
            inp_size = 2305
        elif feature_mode == 'no_dec_top_5':
            inp_size = 1541
            self.linear = whisper_model.decoder.token_embedding
            self.linear.weight.requires_grad = False
        else:
            raise NotImplementedError(f'{feature_mode} not a valid mode: valid modes are emb_only, attn_only, dec_only no_dec, no_prob, no_dec_top_5, all')
        
        # Define layers with specified input and output sizes
        self.fc_layer = nn.Linear(inp_size, n)
        if deep:
            self.fc_layer1 = nn.Linear(inp_size, 2*n)
            self.fc_layer2 = nn.Linear(2*n, n)
        self.output_layer = nn.Linear(n, 1)
        if norm:
            self.inp_norm = nn.LayerNorm(inp_size)
        self.return_logits = return_logits

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        self.mode = feature_mode
        self.norm = norm
        self.deep = deep

    def forward(self, x):
        if self.mode != 'no_prob':
            if x.dim() == 2:
                sm_probs = torch.log(x[:, -1:])
                x = torch.cat((x[:, :-1], sm_probs), dim=1)
            else:
                pass
        
        if self.mode == 'no_dec_top_5':
            dec = x[:, 768:1536]
            attn = x[:, :768]
            emb = x[:, 1536: 2304]
            log_probs = F.log_softmax((dec @ torch.transpose(self.linear.weight.to(dec.dtype), 0, 1)).float(), dim=1)
            top5 = 1 - torch.topk(log_probs, k=5).values
            x = torch.cat((attn, emb, top5), dim=1)

        # Forward pass through each layer
        if self.norm:
            x = self.inp_norm(x)
        if not self.deep:
            x = torch.relu(self.fc_layer(x))
            if x.dim() == 3:
                x, _ = torch.max(x, dim=1)
        else:
            x = torch.relu(self.fc_layer1(x))
            if x.dim() == 3:
                x, _ = torch.max(x, dim=1)
            x = torch.relu(self.fc_layer2(x))

        logits = self.output_layer(x)

        if self.return_logits:
            return logits
        
        # Apply sigmoid activation to get the final output
        output = self.sigmoid(logits)
        
        return output


class TemperatureCalibrator(nn.Module):
    def __init__(self, whisper_model, non_lin: int = 0, temp=1.0):
        super().__init__()
        self.linear = whisper_model.decoder.token_embedding
        self.linear.weight.requires_grad = False
        self.temperature = nn.Parameter(torch.tensor(temp))  
        self.softmax = nn.Softmax(dim=1)
        self.non_lin = non_lin
    
    def forward(self, x):
        x = x[:, :-1]
        with torch.no_grad():
            x = (x @ torch.transpose(self.linear.weight.to(x.dtype), 0, 1)).float()
        x = x * self.temperature
        if self.non_lin:
            x = torch.relu(x)
        sm_probs = self.softmax(x)
        max_prob, _ = torch.max(sm_probs, dim=1)
        return (1 - max_prob).unsqueeze(1) # as we want prob of incorrect

        
class CEMSkip(nn.Module):
    def __init__(self, n: int, feature_mode: str = 'no_dec', return_logits: bool = False,
                 norm: bool = False):
        super().__init__()
        if feature_mode == 'emb_only' or feature_mode == 'attn_only' or feature_mode == 'dec_only':
            inp_size = 769
        elif feature_mode == 'no_dec' or feature_mode == 'no_attn':
            inp_size = 1537
        elif feature_mode == 'no_prob':
            inp_size = 2304
        elif feature_mode == 'all':
            inp_size = 2305
        else:
            raise NotImplementedError(f'{feature_mode} not a valid mode: valid modes are emb_only, no_dec, no_prob, no_attn, all')
        
        # Define layers with specified input and output sizes
        self.fc_layer = nn.Linear(inp_size, n)
        self.output_layer = nn.Linear(n + 1, 1)
        if norm:
            self.inp_norm = nn.LayerNorm(inp_size)

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        self.norm = norm
        self.return_logits = return_logits

    def forward(self, x):
        sm_probs = torch.log(x[:, -1:])

        # Forward pass through each layer
        if self.norm:
            x = self.inp_norm(x)
        x = torch.cat((torch.relu(self.fc_layer(x)), sm_probs), dim=1)
        logits = self.output_layer(x)

        if self.return_logits:
            return logits
        
        # Apply sigmoid activation to get the final output
        output = self.sigmoid(logits)
        
        return output


class ConfidenceDataset(Dataset):
    def __init__(self, pickle_path, feature_mode, pred_mode='token', pool_method='max', prob_comb='max',
                 whisper_model=None, max_tokens_per_word=3):
        if feature_mode in ['emb_only', 'attn_only', 'dec_only', 'no_dec', 'no_dec_top_5', 'no_attn', 'no_prob', 'all']:
            self.feature_mode = feature_mode
        else:
            raise ValueError(f'{feature_mode} not a valid mode: valid modes are emb_only, attn_only, dec_only, no_dec, no_attn, no_prob, all, temp_anneal')

        if pred_mode in ['token', 'word', 'word_new']:
            self.pred_mode = pred_mode
        else:
            raise ValueError('Pred mode must be token or word')
        
        if pool_method in ['max', 'mean', 'first', 'last']:
            self.pool_method = pool_method
        else:
            raise ValueError("Pool method must be one of 'max', 'mean', 'first', 'last'")
        
        if prob_comb in ['max', 'mean', 'prod', 'last', 'inv_prod']:
            self.prob_comb = prob_comb
        else:
            raise ValueError("prob_comb must be one of 'max', 'mean', 'prod', 'inv_prod'")
        

        with open(pickle_path, 'rb') as f:
            word_dicts = pickle.load(f)
            self.attn_features = []
            self.dec_features = []
            self.embs = []
            self.sm_probs = []
            self.targets = []
            max_toks = 1

            for wd in word_dicts:
                if self.pred_mode == 'word':
                    self.attn_features.append(self.feature_pool(wd['attn_features']))
                    self.dec_features.append(self.feature_pool(wd['dec_features']))
                    self.embs.append(self.feature_pool(wd['emb']))
                    self.sm_probs.append([self.agg_probs(wd['sm_probs'])])
                    if len(wd['labels']) == 2:
                        assert(wd['labels'][0] == wd['labels'][1])
                    self.targets.append(wd['labels'][0])
                
                if pred_mode == 'word_new':
                    if len(wd['labels']) > max_tokens_per_word:
                        continue
                    padding = max_tokens_per_word - len(wd['labels'])
                    attn = np.array(wd['attn_features']).astype(np.float32)
                    dec = np.array(wd['dec_features']).astype(np.float32)
                    emb = np.array(wd['emb']).astype(np.float32)

                    if padding:
                        padding_vec = np.zeros((padding, 768), dtype=np.float32)
                        self.attn_features.append(np.concatenate((attn, padding_vec), axis=0))
                        self.dec_features.append(np.concatenate((dec, padding_vec), axis=0))
                        self.embs.append(np.concatenate((emb, padding_vec), axis=0))
                    else:
                        self.attn_features.append(attn)
                        self.dec_features.append(dec)
                        self.embs.append(emb)

                    probs = wd['sm_probs'] + [0.0 for i in range(padding)]
                    self.sm_probs.append([[prob] for prob in probs])
                    self.targets.append(wd['labels'][0])

                
                elif self.pred_mode == 'token':
                    self.attn_features.extend(wd['attn_features'])
                    self.dec_features.extend(wd['dec_features'])
                    self.embs.extend(wd['emb'])
                    self.sm_probs.extend([[prob] for prob in wd['sm_probs']])
                    self.targets.extend(wd['labels'])

        del word_dicts
        gc.collect()

        if self.pred_mode == 'word_new':
            data_type = torch.float16
        else:
            data_type = torch.float32

        self.attn_features = torch.tensor(np.stack(self.attn_features, dtype=np.float32), dtype=data_type)
        self.dec_features = torch.tensor(np.stack(self.dec_features, dtype=np.float32), dtype=data_type)
        self.embs = torch.tensor(np.stack(self.embs, dtype=np.float32), dtype=data_type)
        self.sm_probs = torch.tensor(self.sm_probs, dtype=torch.float32)

        base = 1 if pred_mode=='word_new' else 0

        if feature_mode == 'no_dec':
            self.features = torch.cat((self.attn_features, self.embs, self.sm_probs), axis=1+base)
        elif feature_mode == 'no_attn':
            self.features = torch.cat((self.dec_features, self.embs, self.sm_probs), axis=1+base)
        elif feature_mode == 'all' or feature_mode == 'no_dec_top_5':
            self.features = torch.cat((self.attn_features, self.dec_features, self.embs, self.sm_probs), axis=1+base)
        elif feature_mode == 'emb_only':
            self.features = torch.cat((self.embs, self.sm_probs), axis=1+base)
        elif feature_mode == 'attn_only':
            self.features = torch.cat((self.attn_features, self.sm_probs), axis=1+base)
        elif feature_mode == 'dec_only':
            self.features = torch.cat((self.dec_features, self.sm_probs), axis=1+base)
        elif feature_mode == 'temp_anneal':
            self.features = np.array(self.dec_features)
        elif feature_mode == 'no_prob':
            self.features = torch.cat((self.attn_features, self.dec_features, self.embs), axis=1+base)

        self.num_samples = len(self.attn_features)
        self.shape = self.features.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features = self.features[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return features, target

    def feature_pool(self, arr):
        arr = np.array(arr)
        if len(arr) == 1 or self.pool_method == 'first':
            return arr[0]

        elif self.pool_method == 'max':
            return np.max(arr, axis=0)

        elif self.pool_method == 'mean':
            return np.mean(arr, axis=0)
        
        elif self.pool_method == 'last':
            return arr[-1]
    
    def agg_probs(self, probs):
        if self.prob_comb == 'max':
            return np.max(probs)
        
        if self.prob_comb == 'prod':
            return np.exp(np.sum(np.log(probs))) # do sum of logs for numerical stability

        if self.prob_comb == 'mean':
            return np.mean(probs)
        
        if self.prob_comb == 'inv_prod':
            inv_probs = np.array([1 - prob for prob in probs])
            inv_prod = np.exp(np.sum(np.log(inv_probs)))
            return 1 - inv_prod
    
    def get_x_shape(self):
        return self.features.shape
    
    def get_y_shape(self):
        return self.targets.shape