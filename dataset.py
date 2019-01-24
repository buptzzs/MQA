import torch
import torch.nn as nn
import torchtext
import numpy as np
import os
import json
from torchtext import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

from flair.data import Sentence
import pickle



def preprocess(json_path):
    result = []
    json_data = json.load(open(json_path))
    for d in tqdm(json_data):    
        cur = {}
        cur['id'] = d['id']
        cur['query'] = d['query'].replace('_',' ')
        cur['candidates'] = d['candidates']
        cur['answer'] = d['answer']
        label = -1
        for i in range(len(d['candidates'])):
            if d['candidates'][i] == d['answer']:
                label = i
        assert label != -1
        cur['label'] = label
        cur['supports'] = d['supports']
        result.append(cur)
    return result


class BertField(data.Field):
    
    def __init__(self, tokenizer, fix_length=None):
        super().__init__(pad_token = "[PAD]",fix_length=fix_length, lower = False)
        self.tokenizer = tokenizer
        self.tokenize = self.tokenizer.tokenize
        
    def numericalize(self, batch, device=None):
        batch_seq_ids = []
        if isinstance(batch, tuple):
            batch, length = batch        
        for sentence in batch:
            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence)

            batch_seq_ids.append(torch.tensor(sentence_ids))
        return torch.stack(batch_seq_ids, dim=0)
    
    
class QDataset(data.Dataset):
    '''
    
    '''
    def __init__(self, path, fields, **kwargs):
        make_example = data.example.Example.fromdict
        json_data = json.load(open(path))
        examples = []
        for d in tqdm(json_data):
            examples.append(make_example(d, fields))
            
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(QDataset, self).__init__(examples, fields, **kwargs)
        
class DataHandler:
    
    def __init__(self, train_path, val_path, fields):
        if train_path.endswith('json'):
            print(f'load json data :{train_path}, {val_path}')
            trainset = QDataset(train_path, fields)
            valset = QDataset(val_path, fields)
        else:
            print(f'load examples.pt  :{train_path}, {val_path}')            
            if isinstance(fields, dict):
                fields, field_dict = [], fields
                for field in field_dict.values():
                    if isinstance(field, list):
                        fields.extend(field)
                    else:
                        fields.append(field)            
            train_examples = torch.load(train_path)
            val_examples = torch.load(val_path)            
            trainset = data.Dataset(train_examples, fields)
            valset = data.Dataset(val_examples, fields)
            
        self.trainset = trainset
        self.valset = valset
        
    def get_train_iter(self, batch_size):
        return data.Iterator(self.trainset, batch_size, train=True)
        
    def get_val_iter(self, batch_size):
        return data.Iterator(self.valset, batch_size, train=False, sort=False)

        