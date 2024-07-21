from cem_model import SimpleCEM, ConfidenceDataset, TemperatureCalibrator, CEMSkip
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve as pr_curve, auc as calc_auc
import argparse
import numpy as np
from tools import f_score as calculate_f_score, rel_diagram
from focal_loss import FocalLoss, BinaryFocalLoss
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, pred):
    correct = 0
    for o, p in zip(output.tolist(), pred.tolist()):
        if o>0.5 and int(p)==1 or o<=0.5 and int(p)==0:
             correct+=1
    if output.size(0)==0:
        # import pdb; pdb.set_trace()
        return 100
    else:
        return correct/output.size(0)*100


def train_loop(model, train_loader, criterion, optimizer, temp_anneal=False, backprop=True):
    losses = AverageMeter()
    accs = AverageMeter()
    all_preds = []
    all_tgts = []

    for (features, tgts) in tqdm(train_loader):
        # Compute prediction and loss
        if not temp_anneal:
            features=features.to(device)
            tgts=tgts.to(device)
            pred = model(features)
            loss = criterion(pred, tgts)
        
        else:
            with torch.no_grad():
                features=features.to(device)
                tgts=tgts.to(device)
                pred = model(features)
                loss = criterion(pred, tgts)

        all_preds.extend([p.tolist()[0] for p in pred])
        all_tgts.extend([t.tolist()[0] for t in tgts])

        if backprop:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = accuracy(torch.squeeze(pred),torch.squeeze(tgts))
        accs.update(acc, features.size(0))
        losses.update(loss.item(), features.size(0))
    
    if not temp_anneal:
        print(f'Train loss: {losses.avg}, accuracy: {accs.avg}')
        return losses.avg

    else:
        p, r, _ = pr_curve(all_tgts, all_preds)
        f = calculate_f_score(np.array(p), np.array(r))
        best_f = np.max(f)
        ECE, UCE, accs_rel, count = rel_diagram(all_preds, all_tgts, n=20)
        fig, ax = plt.subplots()
        ax.bar(accs_rel, count, label=f'ECE: {ECE:.5f}, UCE: {UCE:.5f}', width=accs_rel[1]-accs_rel[0], alpha=0.8)
        ax.plot(accs_rel, accs_rel, '--', color='orange', label='perfect calibration') # calibration line
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('True Probability')
        ax.legend()
        
        print(f'Dev loss: {losses.avg}, accuracy: {accs.avg}, AUC: {calc_auc(r, p)}, Best F: {best_f}, UCE: {UCE}, ECE: {ECE}')
        return losses.avg, best_f, calc_auc(r, p), UCE, fig


def temp_anneal(args, hard_code=True):
    # Load whisper model
    whisper_model = whisper.load_model('small.en', device=device, download_root=args.whisper_dir, dropout=0.0, ilm=None, lora=0, 
                                           attention_window=-1, n_ctx=1500, n_text_ctx=448)
    best_ckpt = open(os.path.join(args.whisper_weights_dir, 'best_ckpt')).read().split()[0]
    state_dict = torch.load(f'{args.whisper_weights_dir}/checkpoint/{best_ckpt}')
    state_dict = state_dict['state_dict']
    whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings
    whisper_model = whisper_model.to(device)
    
    # Set up dataloader and temp calibrator model
    dev_dataset = ConfidenceDataset(args.dev_file_path, feature_mode='dec_only')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = ConfidenceDataset(args.test_file_path, feature_mode='dec_only')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    temperature_calibrator = TemperatureCalibrator(whisper_model=whisper_model,
                                                   non_lin=args.non_lin)
    lr = args.initial_lr
    optimizer = torch.optim.Adam(temperature_calibrator.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.BCELoss().to(device)

    if hard_code:
        for temp in np.arange(start=0.94, stop=1.04, step=0.01):
            temperature_calibrator = TemperatureCalibrator(whisper_model=whisper_model,
                                                   non_lin=args.non_lin, temp=temp)
            print(f'---------- Temp {temp} ----------')
            print('Dev')
            loss, best_f, auc, UCE, fig = train_loop(temperature_calibrator, dev_dataloader, criterion, optimizer, temp_anneal=True,
                                                    backprop=False)
            print('Test')
            loss, best_f_test, auc, UCE_test, fig = train_loop(temperature_calibrator, test_dataloader, criterion, optimizer, temp_anneal=True,
                                                    backprop=False)
            test_preds = []
            test_tgts = []
            for (features, tgts) in tqdm(test_dataloader):
                with torch.no_grad():
                    # Compute prediction and loss
                    features=features.to(device)
                    tgts=tgts.to(device)
                    pred = temperature_calibrator(features)
                    test_preds.extend(pred)
                    test_tgts.extend(tgts)
            
            with open(f'exp/cem_models/{args.whisper_model}/temp_cal/{temp}_{best_f_test}.pkl', 'wb') as f:
                pickle.dump((test_preds, test_tgts), f)

            print(f'Temp {temperature_calibrator.temperature}')

    else:
        for epoch in range(args.n_epochs):
            print(f'---------- Epoch {epoch + 1} ----------')
            loss, best_f, auc, UCE, fig = train_loop(temperature_calibrator, dev_dataloader, criterion, optimizer, temp_anneal=True)
            print(f'Temp {temperature_calibrator.temperature}')
    


def test_loop(model, test_loader, criterion):
    losses = AverageMeter()
    accs = AverageMeter()

    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for i, (features, tgts) in enumerate(test_loader):
            # Compute prediction and loss
            features=features.to(device)
            tgts=tgts.to(device)
            pred = model(features)
            loss = criterion(pred, tgts).item()

            acc = accuracy(torch.squeeze(pred),torch.squeeze(tgts))
            accs.update(acc, features.size(0))
            losses.update(loss, features.size(0))

            all_preds.extend([p.tolist()[0] for p in pred])
            all_tgts.extend([t.tolist()[0] for t in tgts])
    
    p, r, _ = pr_curve(all_tgts, all_preds)
    f = calculate_f_score(np.array(p), np.array(r))
    ECE, UCE, accs_rel, count = rel_diagram(all_preds, all_tgts, n=20)
    fig, ax = plt.subplots()
    ax.bar(accs_rel, count, label=f'ECE: {ECE:.5f}, UCE: {UCE:.5f}', width=accs_rel[1]-accs_rel[0], alpha=0.8)
    ax.plot(accs_rel, accs_rel, '--', color='orange', label='perfect calibration') # calibration line
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('True Probability')
    ax.legend()
    best_f = np.max(f)
    print(f'Test loss: {losses.avg}, accuracy: {accs.avg}, AUC: {calc_auc(r, p)}, Best F: {best_f}, UCE: {UCE}')
    return losses.avg, best_f, calc_auc(r, p), UCE, fig


def train(args): 
    print('Loading Datasets')
    train_dataset = ConfidenceDataset(args.train_file_path, feature_mode=args.feature_mode, pred_mode=args.pred_mode,
                                      pool_method=args.pool_mode, prob_comb=args.agg_mode)
    print('Train loaded')
    dev_dataset = ConfidenceDataset(args.dev_file_path, feature_mode=args.feature_mode, pred_mode=args.pred_mode,
                                    pool_method=args.pool_mode, prob_comb=args.agg_mode)
    print('Dev loaded')
    test_dataset = ConfidenceDataset(args.test_file_path, feature_mode=args.feature_mode, pred_mode=args.pred_mode,
                                    pool_method=args.pool_mode, prob_comb=args.agg_mode)
    print('Test loaded')
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_name = args.train_file_path.split('/')[-1].split('.')[0]
    
    if args.BCE_loss:
        outdir = f'exp/cem_models/{args.whisper_model}/{dataset_name}_{args.pred_mode}/{args.hidden_units}_pw{args.pos_weighting}_{args.feature_mode}_norm{args.norm}'
        if args.pos_weighting == 1:
            criterion = nn.BCELoss().to(device)
            return_logits = False
        else:
            pos = args.pos_weighting
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos])).to(device)
            return_logits = True
    
    else:
        outdir = f'exp/cem_models/{args.whisper_model}/{dataset_name}_{args.pred_mode}/{args.hidden_units}_gamma{args.gamma}_{args.feature_mode}_norm{args.norm}'
        criterion = BinaryFocalLoss(gamma=args.gamma).to(device)
        return_logits = False
    
    if args.pred_mode == 'word':
        outdir += f'_pool{args.pool_mode}_prob{args.agg_mode}'

    if args.feature_mode == 'no_dec_top_5':
        whisper_model = whisper.load_model('small.en', device=device, download_root=args.whisper_dir, dropout=0.0, ilm=None, lora=0, 
                                           attention_window=-1, n_ctx=1500, n_text_ctx=448)
        best_ckpt = open(os.path.join(args.whisper_weights_dir, 'best_ckpt')).read().split()[0]
        state_dict = torch.load(f'{args.whisper_weights_dir}/checkpoint/{best_ckpt}')
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict, strict=False) # only load soft prompt embeddings
        whisper_model = whisper_model.to(device)
    
    else:
        whisper_model = None
    
    if args.pretrained_model:
        # logic for loading pretrained model
        raise NotImplementedError
    
    elif not args.prob_skip:
        confidence_model = SimpleCEM(n=args.hidden_units, feature_mode=args.feature_mode, return_logits=return_logits, norm=bool(args.norm),
                                     whisper_model=whisper_model, deep=args.deep).to(device)
    
    else: 
        confidence_model = CEMSkip(n=args.hidden_units, feature_mode=args.feature_mode, return_logits=return_logits, norm=bool(args.norm)).to(device)
        outdir = '/'.join(outdir.split('/')[:-1]) + '/skip' + outdir.split('/')[-1]
    
    if args.deep:
        outdir += '_deep'
    
    if not os.path.exists(outdir):
        # If it doesn't exist, create the directory
        os.makedirs(outdir)
    
    lr = args.initial_lr
    optimizer = torch.optim.Adam(confidence_model.parameters(), lr=lr, weight_decay=0.0)

    train_losses = []
    test_losses = []
    f_arr = []
    UCE_arr = []
    best_f = 0
    best_auc = 0
    best_sum = 0
    iter_no_best = 0

    best_model_paths = {'f_score': None,
                        'auc': None,
                        'sum': None}

    for epoch in range(args.n_epochs):
        print(f'---------- Epoch {epoch + 1} ----------')
        train_loss = train_loop(confidence_model, train_dataloader, criterion, optimizer)
        test_loss, f_score, auc, UCE, fig = test_loop(confidence_model, dev_dataloader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        f_arr.append(f_score)
        UCE_arr.append(UCE)

        checkpoint_path = f'{outdir}/epoch{epoch}_f{f_score:.5f}_loss{test_loss:.3f}_auc{auc:.5f}_UCE{UCE:.4f}.pt'

        if f_score > best_f or auc > best_auc or auc + f_score > best_sum:
            torch.save({
                'epoch': epoch,
                'model_state_dict': confidence_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'f_score': f_score,
                'auc': auc,
                'UCE': UCE,
            }, checkpoint_path)
            
            if f_score > best_f:
                best_model_paths['f_score'] = checkpoint_path
            if auc + f_score > best_sum:
                best_model_paths['sum'] = checkpoint_path
            if auc > best_auc:
                best_model_paths['auc'] = checkpoint_path

            fig.savefig(f'{outdir}/epoch{epoch}_f{f_score:.5f}_loss{test_loss:.3f}_auc{auc:.5f}_UCE{UCE:.4f}.png')
        
        else:
            iter_no_best += 1
        
        best_f = max(best_f, f_score)
        best_auc = max(best_auc, auc)
        best_sum = max(best_sum, f_score + auc)
        
        if iter_no_best >= 4:
            lr /= 10
            iter_no_best = 0
            print(f'Decreasing learning rate to {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if lr < 1e-6:
                break
    
    print(f'Training complete. Best F-score {best_f}. Best auc {best_auc}')
    zipped_metrics = zip(train_losses, test_losses, f_arr, UCE_arr)
    
    # Plot progression of losses with training
    try:
        fig1, ax1a = plt.subplots()
        ax1a.plot(train_losses, label='Train Loss')
        ax1a.plot(test_losses, label='Test Loss')
        ax1a.set_ylabel('Loss')
        plt.legend()
        ax1b = ax1a.twinx()
        ax1b.plot(f_arr, color='green', label='F-Score')
        ax1b.set_ylabel('F-score')
        plt.legend()
        fig1.savefig(f'{outdir}/f_evo.png')

        fig2, ax2a = plt.subplots()
        ax2a.plot(train_losses, label='Train Loss')
        ax2a.plot(test_losses, label='Test Loss')
        ax2a.set_ylabel('Loss')
        plt.legend()
        ax2b = ax2a.twinx()
        ax2b.plot(UCE_arr, color='green', label='UCE')
        ax2b.set_ylabel('UCE')
        plt.legend()
        fig2.savefig(f'{outdir}/UCE_evo.png')

    
    except Exception:
        pass

    with open(f'{outdir}/zipped_metrics.pkl', 'wb') as f:
        pickle.dump(zipped_metrics, f)

    # Delete non-best models and generate test predictions for best models
    for filename in os.listdir(outdir):
        file_path = os.path.join(outdir, filename)
        delete = True
        
        if os.path.isfile(file_path) and 'epoch' in file_path:
            for best_fp in best_model_paths.values():
                if best_fp[:-3] in file_path:
                    delete = False
            if delete:
                os.remove(file_path)
            
            else: # generate test predictions if a best model
                new_fp = file_path.replace('epoch', 'ep')
                os.rename(file_path, new_fp)

                if new_fp.endswith('.pt'): # if model path
                    with torch.no_grad():
                        all_preds = []
                        all_tgts = []

                        confidence_model = SimpleCEM(n=args.hidden_units, feature_mode=args.feature_mode, return_logits=return_logits, 
                                                     norm=bool(args.norm), deep=args.deep).to(device)
                        saved_model = torch.load(new_fp)
                        confidence_model.load_state_dict(saved_model['model_state_dict'])
                        
                        for i, (features, tgts) in enumerate(test_dataloader):
                            # Compute prediction and loss
                            features=features.to(device)
                            tgts=tgts.to(device)
                            pred = confidence_model(features)

                            all_preds.extend([p.tolist()[0] for p in pred])
                            all_tgts.extend([t.tolist()[0] for t in tgts])

                        data = [all_preds, all_tgts]
                        p, r, _ = pr_curve(all_tgts, all_preds)
                        best_f = np.max(calculate_f_score(np.array(p), np.array(r)))
                        auc = calc_auc(r, p)
                        
                        pickle_fp = outdir + '/' + new_fp.split('/')[-1].split('_')[0] + f'testf_{best_f:.4f}_testAUC_{auc:.4f}_preds.pkl'
                        with open(pickle_fp, 'wb') as f:
                            pickle.dump(data, f)

    
if __name__ == '__main__':
    print('Running CEM Train')
    parser = argparse.ArgumentParser(description='Transcribe wav files in a wav list')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to the pretrained model')
    parser.add_argument('--train_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/flt/train_flt__beam5_SA10_30_2_50_2_layer-1_data.pkl', help='path to pickled training dataframe')
    parser.add_argument('--dev_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/flt/dev_flt_beam5_layer-1_data.pkl', help='path to pickled dev dataframe')
    parser.add_argument('--test_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/flt/test_flt_beam5_layer-1_data.pkl')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    parser.add_argument('--hidden_units', default=16, type=int, help='number of hidden units for simple CEM')
    parser.add_argument('--pos_weighting', type=float, default=1.0, help='Weighting to apply to +ve class in weighted BCE loss')
    parser.add_argument('--feature_mode', type=str, default='no_dec', help='Determines which parts of feature vector are to be fed to model: "emb_only", "no_dec", "no_prob", "all"')
    parser.add_argument('--pred_mode', type=str, default='word', help='"word" or "token" - Whether to predict word or token based confidence labels')
    parser.add_argument('--whisper_model', default='flt')
    parser.add_argument('--BCE_loss', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--norm', type=int, default=0, help='Whether to normalise input')
    parser.add_argument('--temp_anneal', type=int, default=0)
    parser.add_argument('--deep', type=int, default=0)
    parser.add_argument('--prob_skip', type=int, default=0)
    parser.add_argument('--pool_mode', type=str, default='max', help='How to go about aggregating token-level features')
    parser.add_argument('--agg_mode', type=str, default='max', help='How to go about aggregating token-level probabilities')
    parser.add_argument('--whisper_dir', type=str, default='/home/mifs/th624/.cache/whisper')
    parser.add_argument('--whisper_weights_dir', type=str, default='exp/small.en/flt_prompt0_lr1e-5_lower', help='output directory to save the transcribed data in jsonl format')
    parser.add_argument('--non_lin', type=int, default=0, help='Whether too allow temp calibrator to apply non-linear activation')

    args = parser.parse_args()
    
    if not args.temp_anneal:
        train(args)
    
    else:
        temp_anneal(args)