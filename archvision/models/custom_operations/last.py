from torch import nn

class LastLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.output = nn.Identity()
                
    def forward(self,x):
        return self.output(x)