import evaluate
from model import BERTPlus
from config import *
from utils import *
from tqdm import tqdm
import argparse
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

def evaluate_model(preds, labels):
    metric = evaluate.load('seqeval')
    all_metrics = metric.compute(predictions=preds, references=labels, zero_division=0.)
    
    overall_metrics = {'precision': all_metrics['overall_precision'], 
                        'recall': all_metrics['overall_recall'],
                        'f1': all_metrics['overall_f1'],
                        'accuracy': all_metrics['overall_accuracy']}
    return overall_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=False, 
        help='path of model checkpoint to be evaluated', type=str,
        default='saved_models/BERTPlus_1_0.001_5')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def main():
    args = parse_args()
    model = BERTPlus(model_checkpoint, model_config).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    torch.set_grad_enabled(False)
    set_seed(seed)

    print('Loading dataset...', flush=True)
    datasets = load_from_disk(test_dataset)
    print('Done!')

    test_dataloader = DataLoader(datasets['test'], collate_fn=collate_batch, 
                                 batch_size=datasets['test'].num_rows, 
                                 shuffle=False, num_workers=2)

    overall_metrics = {'precision':[], 'recall':[], 'f1': [], 'accuracy': []}
    print('Evaluating...')
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(args.device)
        attention_masks = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        loss, logits = model(input_ids, attention_masks, labels)
        # logits = logits.view(labels.size(0), labels.size(1), -1)
        preds = torch.argmax(logits, dim=-1)
        preds, labels = postprocess(preds, labels)
        metrics = evaluate_model(preds, labels)
        for key, val in metrics.items():
            overall_metrics[key].append(val)
    for key, val in overall_metrics.items():
        overall_metrics[key] = np.mean(val)
    print('Evaluation done!')
    print(overall_metrics)


if __name__ == '__main__':
    main()