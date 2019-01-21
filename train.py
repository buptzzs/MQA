import json
import os
import flair
import torch
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, FlairEmbeddings, StackedEmbeddings, CharacterEmbeddings
from QAModel import MultiSentenceEmbeddings, SimpleQANet, CharNgramEmbeddings
from dataset import load_data
from utils import train, val
from flairUtil import BertEmbeddings
from tensorboardX import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
flair.device = torch.device('cuda:0')

class Config:
    
    def __init__(self):
        self.att_dropout = 0.2
        self.rnn_dropout = 0.2
        self.hidden = 100
        self.reproject_words_dimension = None
        self.lr = 1e-3
        self.train_path = './data/qangaroo_v1.1/wikihop/train.pkl'
        self.dev_path = './data/qangaroo_v1.1/wikihop/dev.pkl'
        self.bert_model = './bert-base-uncased/'
        self.bert_token = './bert-base-uncased-vocab.txt'
        self.epochs = 30
        self.log_dir = './logs'
        self.batch_size = 1
        self.model_name = 'SimpleQANet_flair'
config = Config()



save_path = config.model_name + '_epoch'+str(config.epochs) + '_lr'+ str(config.lr)+  \
                '_batchsize' + str(config.batch_size)

save_path = os.path.join(config.log_dir, save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
print(save_path)

writer = SummaryWriter(save_path)

char_emb = CharNgramEmbeddings()
word_emb = WordEmbeddings('en-wiki')

print('build model')
embeddings_list = [word_emb, char_emb
                  # bert_emb, 
                   # f_forward, f_backward
                  ]

embeddings = MultiSentenceEmbeddings(embeddings_list)
net = SimpleQANet(config, embeddings)
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()),
                             lr=config.lr)

criterion = torch.nn.CrossEntropyLoss()

print('loading data')
train_data = load_data(config.train_path)
dev_data = load_data(config.dev_path)

print('train...')
best_acc = 0.0
for epoch in range(config.epochs):
    train_loss, train_acc = train(epoch, train_data, net, criterion, optimizer, batch_size=1, print_period=100)
    val_loss, val_acc = val(epoch, dev_data, net, criterion)
    
    writer.add_scalar('train_loss', train_loss, epoch+1)
    writer.add_scalar('val_loss', val_loss, epoch+1)
    writer.add_scalar('train_acc', train_acc, epoch+1)
    writer.add_scalar('val_acc', val_acc, epoch+1)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))