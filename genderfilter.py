import torch
import torch.nn as nn
from model import regressor
class Genderfilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = regressor().to(self.device).eval()
        ckpt = torch.load('./gender.pt' , map_location=self.device)
        self.net.load_state_dict(ckpt)
        del ckpt
    
    def forward(self, x):
        result = self.net(x)
        return result

