import torch.nn as nn
from transformers import BertModel, AutoModel
from utils import *

class BERTPlus(nn.Module):
    def __init__(self, model_name, config):
        super(BERTPlus, self).__init__()
        if config['use_bert']:
            self.backbone = BertModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        self.cls_head = nn.Linear(config['hidden_size'], config['num_classes'])
        if config['weight_classes']:
            class_weights = compute_class_weights()
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=self.backbone.config.hidden_dropout_prob)
        self.train_count = 0
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.mode = 'linear_probing'

    
    def unfreeze_backbone(self):
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        self.mode = 'finetuning'


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        '''
        input_ids: N (batch size) x T (no. of time steps = seq length)
        attention_mask: N x T
        token_labels: N x T
        '''
        bert_outs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_reps = bert_outs[0] # N x T x H
        if self.mode == 'linear_probing':
            assert bert_hidden_reps.grad_fn is None, 'Computing gradients for BERT'
        else:
            bert_hidden_reps = self.dropout(bert_hidden_reps)
        logits = self.cls_head(bert_hidden_reps) # N x T x C
        # next line shapes: loss_fn((NT x C), NT)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits
