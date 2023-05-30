import torch
from torch import nn
from torchvision.models import mobilenet_v2

class regressor(nn.Module):
    def __init__(self, attrib_num =1):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mobilenet_model = mobilenet_v2(pretrained=True).to(device).eval() # pretrained mobilenet_model
        for params in self.mobilenet_model.parameters():
            params.require_grad = False

        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, attrib_num)      # attr 개수 고려하기
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.mobilenet_model(x) # 8 1000
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return self.act(x) *0.5 + 0.5
