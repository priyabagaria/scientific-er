import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk
import math
import copy
from tqdm import tqdm
from config import *
from utils import *
import argparse
import os
import transformers
from eval import evaluate_model
from model import BERTPlus
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def train(model, train_dataloader, optimizer, device, tb=False, scheduler=None):
    model.train()
    torch.set_grad_enabled(True)
    train_loss = []

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, logits = model(input_ids, attention_masks, labels)
        train_loss.append(loss.detach().item())
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return np.asarray(train_loss).mean()


def validate(model, val_dataloader, device, tb=False):
    model.eval()
    torch.set_grad_enabled(False)
    val_loss = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, logits = model(input_ids, attention_masks, labels)
        val_loss.append(loss.item())
    return np.asarray(val_loss).mean()


def run_epochs(model, num_epochs, train_dataloader, val_dataloader, 
               optimizer, device, tb=False, scheduler=None,
               zeroshot_dataloader=None):
    min_loss = math.inf
    best_model = None
    for epoch in tqdm(range(num_epochs)):
        # train
        train_loss = train(model, train_dataloader, 
                           optimizer, device, scheduler=scheduler)
        print(f'\nEpoch {epoch}, Train Loss: {train_loss}')
        if tb:
            writer.add_scalar('Train/loss', train_loss, model.train_count)
        
        # validate
        val_loss = validate(model, val_dataloader, device)
        if tb:
            writer.add_scalar('Validation/loss', val_loss, model.train_count)
        print(f'Epoch {epoch}, Val Loss: {val_loss}')
        
        # zeroshot validate
        if zeroshot_dataloader is not None:
            zs_loss = validate(model, zeroshot_dataloader, device)
            if tb:
                writer.add_scalar('Validation/zs_loss', zs_loss, model.train_count)
            print(f'Epoch {epoch}, ZS Loss: {zs_loss}')

        # save
        if min_loss > val_loss:
            best_model = copy.deepcopy(model.state_dict())
        model.train_count += 1

    print('Done.')
    return best_model


def evaluate(model, test_dataloader, device):
    model.eval()
    torch.set_grad_enabled(False)
    overall_metrics = {'precision':[], 'recall':[], 'f1': [], 'accuracy': []}
    print('Evaluating...')
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, logits = model(input_ids, attention_masks, labels)
        preds = torch.argmax(logits, dim=-1)
        preds, labels = postprocess(preds, labels)
        metrics = evaluate_model(preds, labels)
        for key, val in metrics.items():
            overall_metrics[key].append(val)
    for key, val in overall_metrics.items():
        overall_metrics[key] = np.mean(val)
    print('Evaluation done!')
    return overall_metrics


def main():
    args = parse_args()
    print_config(args)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    set_seed(seed)

    # data
    print('Loading dataset...', flush=True)
    datasets = load_from_disk(dataset)
    print('Done!')
    print('Train:', datasets.num_rows['train'], 
          'Val:', datasets.num_rows['validation'], 
          'Test:', datasets.num_rows['test'],
          'ZS Val:', datasets.num_rows['zeroshot'])
    params = {
        'collate_fn': collate_batch,
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 2
    }
    train_dataloader = DataLoader(datasets['train'], **params)
    val_dataloader = DataLoader(datasets['validation'], **params)
    zeroshot_dataloader = DataLoader(datasets['zeroshot'], **params)
    test_dataloader = DataLoader(
        datasets['test'], collate_fn=collate_batch, 
        batch_size=datasets['test'].num_rows, shuffle=params['shuffle'], 
        num_workers=params['num_workers']
    )

    # model -- defined with a frozen backbone
    model = BERTPlus(model_checkpoint, model_config).to(args.device)
    if args.cls_weights is not None:
        print(f'Loading weights from {args.cls_weights}')
        state_dict = torch.load(args.cls_weights)
        model.load_state_dict(state_dict)

    # linear probing
    if args.linear_ep > 0:
        optimizer = AdamW(model.parameters(), lr=args.linear_lr)
        best_model = run_epochs(
            model, args.linear_ep, train_dataloader, val_dataloader, 
            optimizer, args.device, tb=args.tb, scheduler=None,
            zeroshot_dataloader=zeroshot_dataloader
        )
        if args.save_models:
            save_fn = get_save_filename(args, linear_probing=True)
            save_path = os.path.join(MODEL_SAVE_PATH, save_fn)
            torch.save(best_model, save_path)

    # finetuning backbone
    if args.finetune_backbone:
        # unfreeze the backbone
        model.unfreeze_backbone()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        # scheduler = transformers  .get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=500,
        #     num_training_steps=args.max_ep*len(train_dataloader),
        # )
        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=args.max_ep*len(train_dataloader),
            lr_end=0,
            power=2.0
        )
        best_model = run_epochs(
            model, args.max_ep, train_dataloader, val_dataloader, 
            optimizer, args.device, tb=args.tb, scheduler=scheduler,
            zeroshot_dataloader=zeroshot_dataloader
        )
        if args.save_models:
            save_path = os.path.join(MODEL_SAVE_PATH, get_save_filename(args))
            torch.save(best_model, save_path)
        print('Model saved.')

    overall_metrics = evaluate(model, test_dataloader, args.device)
    print(overall_metrics)
    print('All done!!!')
    

if __name__ == '__main__':
    main()