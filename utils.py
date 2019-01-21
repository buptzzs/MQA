import random
import torch
import flair

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def train(epoch, data_iter, model, criterion, optimizer, batch_size=1, print_period=100):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()
    random.shuffle(data_iter)
    print('begin training')
    #model.embedding_layer.eval()
    for idx, batch in enumerate(data_iter):
        score = model(batch)
        
        label = batch['label']
        label = torch.LongTensor([label])
        label = label.to(flair.device)
        
        score = score.transpose(0,1)      
        
        loss = criterion(score, label)

        loss = loss / batch_size
        loss.backward()
        if (idx+1)%batch_size == 0 :
            optimizer.step()
            optimizer.zero_grad()        

        losses.update(loss.item())
        
        pred = score.argmax(1)
        acc = pred.eq(label).sum().item()   
        acces.update(acc)
        if (idx) % (batch_size*print_period) == 0:
            print(f'train: epoch:{epoch}, idx:{idx}/{len(data_iter)}, loss:{losses.avg}, acc:{acces.avg}')
    return losses.avg, acces.avg

def val(epoch, data_iter, model, criterion, print_period=100):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()
    print("begin eval")
    for idx, batch in enumerate(data_iter):
        with torch.no_grad():
            score = model(batch)
            
        label = batch['label']
        label = torch.LongTensor([label])
        label = label.to(flair.device)
        
        score = score.transpose(0,1)      
        
        loss = criterion(score, label)
        losses.update(loss.item())
        
        pred = score.argmax(1)
        acc = pred.eq(label).sum().item()   
        acces.update(acc)
        if idx % print_period == 0:
            print(f'val epoch:{epoch}, idx:{idx}/{len(data_iter)}, loss:{losses.avg}, acc:{acces.avg}')
    return losses.avg, acces.avg        