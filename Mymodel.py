import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from resnet import resnet50
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.features = resnet50(pretrained=True)

        self.classifer = nn.Sequential(nn.Linear(2048, 256),
                                       nn.Dropout(0.2),
                                       nn.Linear(256, config.num_class))
    def forward(self, x, meta_data):
        x = self.features(x, meta_data)
        x = self.classifer(x)

        return x
if __name__ == '__main__':
    img = torch.rand([32, 3, 224, 224])
    meta = torch.rand([32, 81])
    model = mymodel()
    out = model(img, meta)
    print(out.size())

