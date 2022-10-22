import re
from venv import create

from datasets import Dataset
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
        ner_tags_by_ids = list(map(lambda label: labels_to_ids[label], list(words_with_labels.values())))
        feature_dict = dict(tokens=list(words_with_labels.keys()),
                    ner_tags=ner_tags_by_ids)
        features.append(feature_dict)
        
    return features

def create_dataset(files):
    feature_dataset = []
    for file in files:
        feature_dataset.extend(load_features_from_file(file))
    dataset = Dataset.from_list(feature_dataset)
    return dataset

dataset = create_dataset(['annotated-data/gpt.conll'])
dataset.save_to_disk('dummydataset.hf')



