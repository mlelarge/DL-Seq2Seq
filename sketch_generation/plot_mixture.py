# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from data_load import get_data
from model import  skrnn_sample
from eval_skrnn import draw_image, load_pretrained_congen, plot_dataset
import torch


data_type = 'kanji'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cond_gen = True
im_nbr = 1000


def plot_mixture(stroke, mixture_params):
    delta = 1000
    x = np.cumsum(stroke[:, 0], 0)
    y = -np.cumsum(stroke[:, 1], 0)

    x_lin = np.linspace(x.min()-0.3, x.max()+0.3, delta)
    y_lin = np.linspace(y.min()-0.3, y.max()+0.3, delta)
    X,Y = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(X)
    grid = np.dstack([X,Y])
    
    """Compute the pdf for the whole sketch i.e. sum for all the strokes"""
    for i in range(len(mixture_params)):
        mu = [x[i], y[i]]
        C = np.zeros((2,2)) 
        C[0,0], C[1,1] = mixture_params[i,2]*mixture_params[i,2], mixture_params[i,3]*mixture_params[i,3]
        C[0,1], C[1,0] = mixture_params[i,4]*mixture_params[i,2]*mixture_params[i,3], mixture_params[i,4]*mixture_params[i,2]*mixture_params[i,3]
        Z+= multivariate_normal(mean = mu, cov=C).pdf(grid)

    """Clipping the pdf to have a nice plot (avoid scaling effect)"""
    Z = Z.clip(0.0, 10.0)
    
    """plot the pdf"""
    fig, ax = plt.subplots(1,1)
    cs = ax.contourf(x_lin, y_lin, Z)
    fig.colorbar(cs, ax=ax)
    plt.axis('off')
    plt.savefig('drawings/mixtures/mixture_'+data_type+'_'+str(im_nbr)+'.png', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    
    data_enc, data_dec, max_seq_len = get_data(data_type = data_type, max_len=200)
    inp_enc =  torch.tensor(data_enc[im_nbr], dtype=torch.float, device=device).unsqueeze(0)
    encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_congen(data_type)

    """plotting the original sketch, then compute and plot the generated sketch"""
    plot_dataset(data_enc, im_nbr, save=True, save_dir = "drawings/mixtures/",
                 name = 'original_'+data_type+'_'+str(im_nbr))
    
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim,
                                time_step=max_seq_len, random_state = 700, cond_gen=cond_gen, 
                                device=device, bi_mode= mode,inp_enc = inp_enc, temperature = 0.2)
    
    draw_image(strokes, save = True, save_dir = "drawings/mixtures/",
               name = 'decoded_'+data_type+'_'+str(im_nbr))
    
    """plot the mixture pdf of the generated sketch"""
    plot_mixture(strokes, mix_params)
