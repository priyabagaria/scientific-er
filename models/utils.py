import torch
import argparse
import random, os
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from config import *

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear-ep', type=int, default=0,
                        help='number of epochs for training with frozen backbone')
    parser.add_argument('--max-ep',
                        help='count of maximum number of epochs for training'
                        ' while finetuning the backbone', 
                        type=int)
    parser.add_argument('--save-models', action='store_true',
                        help='save model with least validation loss to memory')
    parser.add_argument('--linear-lr', type=float,
                        help='learning rate for training with frozen backbone')
    parser.add_argument('--lr', type=float,
                        help='learning rate for training while finetuning backbone')
    parser.add_argument('--batch-size',
                        help='batch size for training and validation', type=int)
    parser.add_argument('--tb', help='use tensorboard', action='store_true')
    parser.add_argument('--finetune-backbone', 
                        help='finetune bb while training', action='store_true')
    parser.add_argument('--cls_weights', type=str, default=None,
                        help='file path to weights file')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def print_config(args):
    print('='*20)
    print('--CONFIGURATION--')
    print(f'Device: {args.device}')
    print(f'Epoch count: {args.max_ep}')
    print(f'Linear Probing Epoch count: {args.linear_ep}')
    print(f'Learning rate: {args.lr}')
    print(f'Linear Probling Learning rate: {args.linear_lr}')
    print(f'Batch size: {args.batch_size}')
    print(f'Save models: {args.save_models}')
    print(f'Finetune backbone: {args.finetune_backbone}')
    print('='*20)


def set_seed(seed):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def collate_batch(batch):
    ds = data_collator([batch[i] for i in range(len(batch))])
    return ds


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels


def get_save_filename(args, linear_probing=False):
    path = 'BERTPlus'
    if model_config['use_bert']:
        path += '_bert'
    else:
        path += '_scibert'
    path += f'_{args.batch_size}'
    if args.finetune_backbone:
        path += '_ftnebb'
    if args.cls_weights is not None:
        path += '_ldcls'
    if model_config['weight_classes']:
        path +=  '_wc'
    if linear_probing:
        path += f'_{args.linear_ep}'
        path += f'_{args.linear_lr}'
        path += '_linear_probing'
    else:
        path += f'_{args.max_ep}'
        path += f'_{args.lr}'
    return path


def compute_class_weights():
    data = load_from_disk(dataset)
    train_labels = [
                    token_label for seq_label in data['train']['labels'] 
                    for token_label in seq_label
                    ]
    # Ignoring the labels of -100
    class_sample_count = np.unique(train_labels, return_counts=True)[1][1:] 
    class_weights = torch.from_numpy(1 / class_sample_count).float()
    return class_weights