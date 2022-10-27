## Train Extensions of BERT/SciBERT

This repository contains the model definition for `BERTPlus` which allows 
training variants of BERT and SciBERT by learning a classifier head and/or
finetuning the backbone BERT/SciBERT.

To generate the dataset using annotated `CoNLL` files:
1. Specify annotation lists (paths to annotations) in `config.py` for generating 
data (if needed) and the final dataset name to save the `DatasetDict` in the current 
directory
2. You can optionally specify `gen_test_dataset=True` in `config.py` to generate a 
separate dataset for testing only, using `config.test_annotations_list`
3. Adjust `split_ratios` for a custom train-val-test split. By default, 75% of the data
is in the train split, 20% is validation, 5% test
3. To generate the dataset according to specifications in the `config.py` file, run
`python gen_dataset.py`

To train:
1. Specify `model_checkpoint` (Eg. 'bert-base-cased') in `config.py` and `use_bert`
in `config.model_config` as `True` if using BERT as backbone, and `False` if using
SciBERT as backbone
2.  Run `train.py` with the following arguments (as necessary):
```
python train.py --linear-ep LIN_EP --max-ep MAX_EP --save-models --linear-lr LIN_LR
--lr LR --batch-size BS --tb --finetune-backbone
```

|Argument|Description|
|---|---|
|`--save-models` | save model with least validation loss to memory |
|`--linear-lr` | learning rate for training with frozen backbone |
| `--lr` | learning rate for training while finetuning backbone |
| `--batch-size` | batch size for training and validation |
| `--tb` | use tensorboard |
| `--finetune-backbone` | finetune bb while training |
| `--cls_weights` | file path to weights file |

3. You can optionally use `run.sh` which has existing commands

To test on a different paper/list of papers (test dataset):
Run:
```
python eval.py --ckpt-path MODEL_CHECKPOINT_PATH
```

