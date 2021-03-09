import matplotlib.pyplot as plt
import numpy as np
from data_load import get_data
from model import skrnn_loss
from eval_skrnn import load_pretrained_congen, load_pretrained_uncond
import torch

data_types = ['cat', 'cake']
weights = [0.25, 0.5, 1.0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=50

losses = np.zeros((len(data_types), len(weights),4))

for i, data_type in enumerate(data_types):
    data_enc, data_dec, max_seq_len = get_data(data_type = data_type, max_len=200, train_mode='test')
    num_mini_batch = len(data_dec) - (len(data_dec) % batch_size)

    for j, weight_kl in enumerate(weights):

        sum_ltot, sum_lr, sum_lkl = 0.0, 0.0, 0.0  
        encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_congen(data_type, weight_kl)    
           
        for batch_id in range(0, num_mini_batch, batch_size):
            hidden_enc = hidden_dec = encoder.initHidden()
            
            inp_enc = torch.tensor(data_enc[batch_id:batch_id+batch_size], dtype=torch.float, device=device)
            inp_dec = torch.tensor(data_dec[batch_id:batch_id+batch_size], dtype=torch.float, device=device)
                
            z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)  
            gmm_params, _ = decoder(inp_dec, z, hidden_dec)
                   
            loss_lr, loss_kl = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
            sum_ltot += (loss_lr + weight_kl*loss_kl).cpu().detach().numpy()
            sum_lr += loss_lr.cpu().detach().numpy()
            sum_lkl += loss_kl.cpu().detach().numpy()
           
        losses[i,j,0] = sum_ltot/num_mini_batch
        losses[i,j,1] = sum_lr/num_mini_batch
        losses[i,j,2] = sum_lkl/num_mini_batch


for i, data_type in enumerate(data_types):
    data_enc, data_dec, max_seq_len = get_data(data_type = data_type, max_len=200, train_mode='test')
    num_mini_batch = len(data_dec) - (len(data_dec) % batch_size)
    weight_kl = 0.0
    
    sum_ltot, sum_lr, sum_lkl = 0.0, 0.0, 0.0  
    encoder, decoder, hidden_size, latent_dim, cond_gen, mode, device = load_pretrained_uncond(data_type) 

    for batch_id in range(0, num_mini_batch, batch_size):
        hidden_dec = (torch.zeros(1, 50, hidden_size, device=device), torch.zeros(1, 50, hidden_size, device=device))

        z = torch.zeros(50, latent_dim, device=device)

        gmm_params, _ = decoder(inp_dec, z, hidden_dec)
        
        loss_lr, loss_kl = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
        sum_ltot += (loss_lr + weight_kl*loss_kl).cpu().detach().numpy()
        sum_lr += loss_lr.cpu().detach().numpy()
        sum_lkl += loss_kl.cpu().detach().numpy()

    losses[i,:,3] = sum_ltot/num_mini_batch


fig, ax = plt.subplots(2,2, figsize = (15,15))

ax[0,0].plot(weights, losses[0,:,0], color = 'orange', marker = '+')
ax[0,0].axhline(losses[0,0,3], color = 'orange', linestyle = '--')
ax[0,0].plot(weights, losses[1,:,0], color = 'blue', marker = '+')
ax[0,0].axhline(losses[1,0,3], color = 'blue', linestyle = '--')
ax[0,0].legend(["Cat cond", "Cat uncond", "Cake cond", "Cake uncond"])
ax[0,0].set_xlabel('Weight KL')
ax[0,0].set_ylabel('LR + LKL')


ax[0,1].plot(weights, losses[0,:,1], color = "orange", marker = '+')
ax[0,1].axhline(losses[0,0,3], color = 'orange', linestyle = '--')
ax[0,1].plot(weights, losses[1,:,1], color = "blue", marker = '+')
ax[0,1].axhline(losses[1,0,3], color = 'blue', linestyle = '--')
ax[0,1].legend(["Cat cond", "Cat uncond", "Cake cond", "Cake uncond"])
ax[0,1].set_xlabel('Weight KL')
ax[0,1].set_ylabel('LR')

ax[1,0].plot(weights, losses[0,:,2], color = 'orange', marker = '+')
ax[1,0].plot(weights, losses[1,:,2], color = 'blue', marker = '+')
ax[1,0].legend(["Cat cond", "Cake cond"])
ax[1,0].set_xlabel('Weight KL')
ax[1,0].set_ylabel('LKL')

ax[1,1].plot(weights, losses[0,:,1], color = 'orange', marker = '+')
ax[1,1].axhline(losses[0,0,3], color = 'orange', linestyle = '--')
ax[1,1].plot(weights, losses[1,:,2], color = 'blue', marker = '+')
ax[1,1].axhline(losses[1,0,3], color = 'blue', linestyle = '--')
ax[1,1].legend(["Cat cond", "Cat uncond", "Cake cond", "Cake uncond"])
ax[1,1].set_xlabel('LKL')
ax[1,1].set_ylabel('LR')

plt.savefig("plot_loss.png")
plt.show()
