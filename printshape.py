import torch
#import pandas as pd
import numpy as np
import torchvision.models as models
from torchsummary import summary
 
model = models.resnet18(pretrained=False)
#model1=models.BasicBlock()
#model1.cuda()
model=model.cuda()
x=torch.randn(1,6,500,500)

layer1out = model.forward(x.cuda()).data
print(layer1out.shape)

#parm={}
#for name,parameters in model.named_parameters():
    #print(name,':',parameters.size())
    #parm[name]=parameters.detach().numpy()