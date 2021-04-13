import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons
from model import build_model,update_generator,update_discriminator
from loss_function import discriminator_loss,generator_loss
import torch
import os
import torch.optim as optim

'''describe
moon data fitting
input normal noise z=(z1,....z_d) ---> output 2d data x=(x1,x2)

plt.scatter(x[:,0],x[:,1],color='cyan',edgecolor='black',marker='s',s=40)
plt.show()
'''

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def draw_data(x,batch_size,noise_dim):
    #random normal noise d dimension
    n_sample=x.shape[0]
    z=torch.randn(batch_size,noise_dim)
    batch_idx=np.random.choice(range(n_sample),size=batch_size).tolist()
    x_real=x[batch_idx,:]
    return z,x_real


def train(data,noise_dim,epochs,lr,k_step_update,batch_size):
    data_dim=data.shape[-1]
    G=build_model(input_size=noise_dim,output_size=data_dim)
    D=build_model(input_size=data_dim,output_size=1,discriminator=True)

    d_optim=optim.Adam(D.parameters(),lr=lr)
    g_optim=optim.Adam(G.parameters(),lr=lr)

    g_loss_fnc=generator_loss()
    d_loss_fnc=discriminator_loss()

    g_loss='no update'


    i=1
    print('start train...')
    plt.ion()
    for ep in range(1,epochs+1):

        #draw and update discriminator
        noise, real = draw_data(data,batch_size,noise_dim)
        d_loss=update_discriminator(noise,real,D,G,d_loss_fnc,d_optim)

        #if in k step: update generator
        if i==k_step_update:
            noise, real = draw_data(data,batch_size,noise_dim)
            g_loss=update_generator(noise,D,G,g_loss_fnc,g_optim)
            i=1
        else:
            i+=1

        if ep%5==0:
            noise = torch.randn(n_sample*2, noise_dim)

            plt.cla()
            plt.scatter(G(noise).data.numpy()[:,0],G(noise).data.numpy()[:,1], label=f'Generated data(size:{n_sample*2})',
                        color='red',edgecolor='black',marker='s',s=40)
            plt.scatter(x[:,0],x[:,1] ,label=f'real data(size:{n_sample})',color='cyan',edgecolor='black',marker='s',s=20)

            plt.text(-.3, 2.3, 'D(x)=%.2f (0.5 for D to converge)' % D(real).mean().item(),fontdict={'size': 13})
            plt.text(-.3, 2, 'D(G(z))= %.2f (0.5 for G to converge)' % D(G(noise)).mean().item(), fontdict={'size': 13})
            plt.legend(loc='lower left', fontsize=10)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(-1.5,2.5)
            plt.ylim(-2.5,2.5)
            plt.title('GAN learn moon data distribution')
            plt.draw()
            plt.pause(0.01)
    plt.ioff()
    plt.show()


if __name__=='__main__':
    n_sample = 1000
    x, y = make_moons(n_samples=n_sample, noise=0.2, random_state=0)
    x = torch.tensor(x, dtype=torch.float32)
    train(data=x, noise_dim=10, epochs=5000, lr=0.001, k_step_update=3, batch_size=64)