from torch import nn
from kymatio.torch import Scattering2D
import torch

class ScatTransform(nn.Module):
    
    def __init__(self, C:int, J: int,L: int,M: int,N: int) -> None:
        
        super(ScatTransform, self).__init__()
        
        self.C, self.J, self.L, self.M,self.N =  C, J, L, M, N
        self.model = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L)  
        self.channel_size = int(self.C * (1 + (self.L * self.J) + (((self.L**2*self.J)*(self.J-1))/2)))
        self.layer_size = int(self.channel_size * (self.M/(2**self.J))  * (N/(2**self.J)))

    
    def forward(self, x:nn.Module) -> torch.Tensor:
        out = self.model(x)
        #print(out.shape)
        return out