from cem_model import SimpleCEM, CEMSkip, ConfidenceDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve as pr_curve, auc 
import argparse
import numpy as np
from tools import f_score
from tqdm import tqdm
import pickle
import os

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


def train_loop(model, train_loader, criterion, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()

    for (features, tgts) in tqdm(train_loader):
        # Compute prediction and loss
        features=features.to(device)
        tgts=tgts.to(device)
        pred = model(features)
        loss = criterion(pred, tgts)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(torch.squeeze(pred),torch.squeeze(tgts))
        accs.update(acc, features.size(0))
        losses.update(loss.item(), features.size(0))
    
    print(f'Train loss: {losses.avg}, accuracy: {accs.avg}')
    return losses.avg


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
    f = f_score(np.array(p), np.array(r))
    best_f = np.max(f)
    print(f'Test loss: {losses.avg}, accuracy: {accs.avg}, AUC: {auc(r, p)}, Best F: {best_f}')
    return losses.avg, best_f, auc(r, p)


def train(args): 
    if args.use_dec==0:
        no_dec=True
    else:
        no_dec=False
    
    print('Loading Datasets')
    train_dataset = ConfidenceDataset(args.train_file_path, no_dec=no_dec)
    print('Train loaded')
    dev_dataset = ConfidenceDataset(args.dev_file_path, no_dec=no_dec)
    print('Dev loaded')
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    if args.pretrained_model:
        # logic for loading pretrained model
        pass
    else:
        if args.prob_skip:
            confidence_model = CEMSkip(n=args.hidden_units, no_dec=no_dec).to(device)
        else:
            confidence_model = SimpleCEM(n=args.hidden_units, no_dec=no_dec).to(device)
    
    if not os.path.exists(args.outdir):
        # If it doesn't exist, create the directory
        os.makedirs(args.outdir)
    
    criterion = nn.BCELoss().to(device)
    lr = args.initial_lr
    optimizer = torch.optim.Adam(confidence_model.parameters(), lr=lr, weight_decay=0.0)

    train_losses = []
    test_losses = []
    f_arr = []
    best_f = 0
    iter_no_best = 0

    for epoch in range(args.n_epochs):
        print(f'---------- Epoch {epoch + 1} ----------')
        train_loss = train_loop(confidence_model, train_dataloader, criterion, optimizer)
        test_loss, f_score, auc = test_loop(confidence_model, test_dataloader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        f_arr.append(f_score)

        checkpoint_path = f'{args.outdir}/ckpt_epoch{epoch}_f{f_score:.5f}_loss{test_loss:.3f}_auc{auc:.5f}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': confidence_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'f_score': f_score,
            'auc': auc
        }, checkpoint_path)

        if f_score > best_f:
            best_f = f_score
            iter_no_best = 0
        
        else:
            iter_no_best += 1
        
        if iter_no_best >= 4:
            lr /= 10
            iter_no_best = 0
            print(f'Decreasing learning rate to {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if lr < 1e-5:
                break
    
    print(f'Training complete. Best F-score {best_f}')
    zipped_metrics = zip(train_losses, test_losses, f_arr)
    with open(f'{args.outdir}/zipped_metrics.pkl', 'wb') as f:
        pickle.dump(zipped_metrics, f)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe wav files in a wav list')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to the pretrained model')
    parser.add_argument('--train_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/train_beam5_SA5_22_2_0_0.pkl', help='path to pickled training dataframe')
    parser.add_argument('--dev_file_path', type=str, default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/exp/cem_data/dev_beam5.pkl', help='path to pickled dev dataframe')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    parser.add_argument('--no_prob', type=int, default=0)
    parser.add_argument('--hidden_units', type=int, help='number of hidden units for simple CEM')
    parser.add_argument('--outdir', type=str, help='path to save model data')
    parser.add_argument('--use_dec', type=int, default=1, help='whether to use decoder state in feature vector')
    parser.add_argument('--prob_skip', type=int, default=0, help='whether to provide skip connection around hidden layer for softmax prob')
    args = parser.parse_args()

    train(args)