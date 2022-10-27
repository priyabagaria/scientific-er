from glob import glob

label_names = ["O", "B-MethodName", "I-MethodName", "B-HyperparameterName",
        "I-HyperparameterName", "B-HyperparameterValue", "I-HyperparameterValue",
        "B-MetricName", "I-MetricName", "B-MetricValue", "I-MetricValue",
        "B-TaskName", "I-TaskName", "B-DatasetName", "I-DatasetName"]
id2label = {i:label for i, label in enumerate(label_names)}
label2id = {label:i for i, label in enumerate(label_names)}
# model_checkpoint = "bert-base-cased"
model_checkpoint = "allenai/scibert_scivocab_uncased"
num_classes = len(label_names)
annotations_list = glob('../annotations/*')
zero_shot_annotations_list = ['../test_annotations/emsum.conll']
test_annotations_list = ['../test_annotations/kg-nli.conll']
seed = 1234
split_ratios = [0.25, 0.20]
test_split_ratios = [0.99, 0.99]
gen_test_dataset = True
MAX_SEQ_LEN = 128
model_config = {'hidden_size': 768, 'num_classes': num_classes, 'weight_classes': False, 'use_bert': False}
# model_checkpoint = 'bert-base-cased'
dataset = 'tokenized_dataset_scibert'
test_dataset = 'test_dataset_scibert'
MODEL_SAVE_PATH = '../saved_models'
