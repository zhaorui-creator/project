import torchvision
#import torchvision.models as models
from torchvision.models import resnet18
from torchsummary import summary
import torch
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from data.testdataset import testdataset
import visdom
import os
from collections import OrderedDict

device=torch.device("cuda")

def evalute(model,loader):
    correct=0
    total=len(loader.dataset)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            model.eval()
            logits=model(x)
            pred=logits.argmax(dim=1)
            print(logits)
            print(pred)
        #visual
        
        correct+=torch.eq(pred,y).sum().float().item()
    
    return correct/total


def test():
    #data
    test_data=testdataset('/home/luo/rui/classification/test1','test')
    test_loader=DataLoader(test_data,batch_size=32,shuffle=False,num_workers=4)
    result=[]
    #model
    model=resnet18(pretrained=False)
    path_state_dict='/home/luo/rui/classification/checkpoints/checkpoint_model_epoch_34_loss0.7452232837677002.pth.tar'
    
    state_dict_load=torch.load(path_state_dict)

    new_state_dict=OrderedDict()
    for k,v in state_dict_load.items():
        namekey=k[7:] if k.startswith('module.') else k
        new_state_dict[namekey]=v

    model.load_state_dict(new_state_dict)
    model.to(device)

    print('loaded from ckpt:')
    test_acc=evalute(model,test_loader)
    print('test_acc:',test_acc)

if __name__=='__main__':
    test()