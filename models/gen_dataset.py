from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from config import *

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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
        ner_tags_by_ids = list(map(lambda label: label2id[label], 
                                list(words_with_labels.values())))
        tokens = list(words_with_labels.keys())
        if tokens == []:
            continue
        feature_dict = dict(tokens=tokens, ner_tags=ner_tags_by_ids)
        features.append(feature_dict)
        
    return features


def create_dataset(files):
    feature_dataset = []
    for f in files:
        feature_dataset.extend(load_features_from_file(f))
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
        remove_columns=datasets[list(datasets.keys())[0]].column_names,
        )
    return tokenized_datasets


def generate_dataset(paper_list, test=True):
    global split_ratios
    dataset = create_dataset(paper_list)
    zero_shot_dataset = create_dataset(zero_shot_annotations_list)
    if test:
        split_ratios = test_split_ratios
    dataset_dict = dataset.train_test_split(test_size=split_ratios[0], seed=seed) 
    test_val_dataset_dict = dataset_dict['test'].train_test_split(test_size=split_ratios[1], seed=seed)

    dataset_dict = DatasetDict({
      'train': dataset_dict['train'],
      'validation': test_val_dataset_dict['train'],
      'test': test_val_dataset_dict['test'],
      'zeroshot': zero_shot_dataset
    })
    if test:
        dataset_dict = DatasetDict({
            'test': test_val_dataset_dict['test']
        })
    tokenized_datasets = preprocess_dataset(dataset_dict)
    return tokenized_datasets


def main():
    global annotations_list, dataset
    if gen_test_dataset:
        annotations_list = test_annotations_list
        dataset = test_dataset
    datasets = generate_dataset(annotations_list, test=gen_test_dataset)
    # datasets = generate_dataset(['models/annotations/mtmt.conll']) ## for debugging
    datasets.save_to_disk(dataset)

    # train_dataloader = DataLoader(datasets['train'], collate_fn=collate_batch, batch_size=5, shuffle=False)
    # print(next(iter(train_dataloader)))

if __name__ == '__main__':
    main()