import torch.nn as nn
import torch

class discriminator_loss(nn.Module):
    def __init__(self):
        super(discriminator_loss, self).__init__()

    def forward(self,p_data,p_g):
        #minimize entropy of D(x)---> maximize D(x)
        #minimize entropy of 1-D(G(z)) ----> minimize D(G(z))
        term_1=torch.log(p_data)
        term_2=torch.log(1-p_g)
        loss=-torch.mean(term_1+term_2)
        return loss


class generator_loss(nn.Module):
    def __init__(self):
        super(generator_loss, self).__init__()

    def forward(self,p_g):
        #minimize negative entropy of 1-D(G(z)) --->maximize D(G(z))
        loss=torch.mean(torch.log(1-p_g))
        return loss
