from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
SEQ_MAX_LENGTH=512
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                padding=True, max_length=SEQ_MAX_LENGTH)
# entity_labels = ["B-MethodName", "I-MethodName", "B-HyperparameterName",
#         "I-HyperparameterName", "B-HyperparameterValue", "I-HyperparameterValue",
#         "B-MetricName", "I-MetricName", "B-MetricValue", "I-MetricValue",
#         "B-TaskName", "I-TaskName", "B-DatasetName", "I-DatasetName"]
entity_labels = ['B-DatasetName', 'B-HyperparameterName', 'B-HyperparameterValue', 
        'B-MethodName', 'B-MetricName', 'B-MetricValue', 'B-TaskName', 
        'I-DatasetName', 'I-HyperparameterName', 'I-HyperparameterValue', 
        'I-MethodName', 'I-MetricName', 'I-MetricValue', 'I-TaskName', 'O']

ids_to_labels = {id: label for id, label in enumerate(entity_labels)}
labels_to_ids = {label: id for id, label in enumerate(entity_labels)}


"""
    This expects a file in the CONLL format
"""
def load_features_from_file(conll_file):
    text = ''
    with open(conll_file, 'r') as f:
        text = f.read()
    
    text = text.replace('-X- _ ', '').replace('-X- ', '')
    
    sentences = text.split('\n\n')

    features = []

    for sentence in sentences:
        lines = list(filter(len, sentence.split('\n')))
        
        words_with_labels = {line.split()[0]: line.split()[1] for line in lines}
        ner_tags_by_ids = list(map(lambda label: labels_to_ids[label], 
                                list(words_with_labels.values())))
        tokens = list(words_with_labels.keys())
        if tokens == []:
            continue
        feature_dict = dict(tokens=tokens,
                    ner_tags=ner_tags_by_ids)
        features.append(feature_dict)
        
    return features

def create_dataset(files):
    feature_dataset = []
    for file in files:
        feature_dataset.extend(load_features_from_file(file))
    dataset = Dataset.from_list(feature_dataset)
    return dataset 


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(data):
    tokenized_inputs = tokenizer(
        data["tokens"], truncation=False, is_split_into_words=True
    )
    all_labels = data["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def preprocess_dataset(datasets):
    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=datasets["train"].column_names,
        )
    return tokenized_datasets

def collate_batch(batch):
    print(batch)
    ds = data_collator([batch[i] for i in range(len(batch))])
    return ds
    

dataset = create_dataset(['annotated-data/gpt.conll'])
dataset.save_to_disk('dummydataset.hf')

dataset = load_from_disk('dummydataset.hf')
dataset_dict = dataset.train_test_split(test_size=0.35, seed=5) 
tokenized_datasets = preprocess_dataset(dataset_dict)


train_dataloader = DataLoader(tokenized_datasets['train'], collate_fn=collate_batch, batch_size=5, shuffle=False)
print(next(iter(train_dataloader)))
