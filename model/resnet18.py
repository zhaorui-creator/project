import torchvision.models as models
from torchsummary import summary
import torch
backbone=models.resnet50(pretrained=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mymodel=backbone.to(device)


summary(mymodel,(3,500,500))
#print(torch.cuda.is_available())




