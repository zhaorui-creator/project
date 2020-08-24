
#from collections import OrderedDict
#import torch
#path_state_dict='./best8.mdl'
        
#state_dict_load=torch.load(path_state_dict)

#new_state_dict=OrderedDict()
#for k,v in state_dict_load.items():
    #namekey=k[7:] if k.startswith('module.') else k
    #new_state_dict[namekey]=v
    #print(namekey)


import torch
import torchvision
from torchsummary import summary          #使用 pip install torchsummary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18()
print(model)