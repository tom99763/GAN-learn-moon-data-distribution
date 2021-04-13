import torch
import torch.nn as nn

def build_model(input_size,output_size,discriminator=False):
    if discriminator:
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    else:
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    return model


def update_discriminator(noise,real,D,G,d_loss_fnc,d_optim):
    fake=G(noise).detach()  #這時候的G是fixed,不更新Generator
    d_loss=d_loss_fnc(D(real),D(fake))
    d_optim.zero_grad()
    d_loss.backward(retain_graph=True) #gradient ascend
    d_optim.step()
    return d_loss.item()

def update_generator(noise,D,G,g_loss_fnc,g_optim):
    fake=G(noise)
    g_loss=g_loss_fnc(D(fake))
    g_optim.zero_grad()
    g_loss.backward()
    g_optim.step()
    return g_loss.item()
