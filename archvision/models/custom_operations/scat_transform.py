from torch import nn
from kymatio.torch import Scattering2D
import torch

class ScatTransform(nn.Module):
    
    def __init__(self,
                J: int,
                L: int,
                M: int,
                N: int):
        
        super(ScatTransform, self).__init__()
        
        self.J, self.L, self.M,self.N =  J, L, M, N
        self.model = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L)  
        self.layer_size = int(3 * (1 + (self.L * self.J) + (((self.L**2*self.J)*(self.J-1))/2)) * (self.M/(2**self.J))  * (N/(2**self.J)))

    
    def forward(self, x:nn.Module):
        return self.model(x)