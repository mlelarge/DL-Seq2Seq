from data_load import get_data
from model import skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond, load_pretrained_congen, plot_dataset
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

data_type = 'cake' # can be kanji character or cat or cake
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cond_gen = True


seeds = np.random.randint(0, 2500, 8)
temperatures = np.linspace(0.01, 1.0, 10)
color =  cm.get_cmap('brg', len(temperatures)*2)

"""Conditional generation"""
if cond_gen == True:
    
    data_enc, data_dec, max_seq_len = get_data(data_type = data_type, max_len=200)
    encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_congen(data_type)
    
    """For each seed and temperature, generate a sketch and save the figure"""
    for j, seed in enumerate(seeds):
        inp_enc = torch.tensor(data_enc[seed], dtype=torch.float, device=device).unsqueeze(0)
        name = "original_"+str(seed)
        plot_dataset(data_enc, seed, save = True, save_dir = 'drawings/conditional/'+data_type+'/',
                 name = 'original_'+str(seed), color='green')
    
        for i, temp in enumerate(temperatures):
            name = str(temp) + "_" + str(seed)     
            strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim,
                            time_step= max_seq_len, random_state = seed, cond_gen=True, 
                            device=device, bi_mode= mode, temperature = temp, inp_enc = inp_enc)
            draw_image(strokes,save=True,save_dir='drawings/conditional/'+data_type+'/', color = color(i),
            name = name)

    """Load the saved figures and plot them all in a common frame along with the original sketches"""
    fig, ax = plt.subplots(11, 8, figsize = (30,30))

    for j, seed in enumerate(seeds):
        name = "original_" + str(seed)            
        im = plt.imread('drawings/conditional/'+data_type+'/'+name+'.png')
        
        ax[0, j].axis('off')
        ax[0, j].imshow(im)
        
        for i, temp in enumerate(temperatures):
            
            name = str(temp) + "_" + str(seed)            
            im = plt.imread('drawings/conditional/'+data_type+'/'+name+'.png')
            ax[i+1,j].axis('off')
            ax[i+1,j].imshow(im)

    plt.axis('off')
    plt.savefig('drawings/conditional/'+data_type+'/complete.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()

    """Unconditional generation"""

else : 
    _, _, max_seq_len = get_data(data_type = data_type, max_len=200)
    encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_uncond(data_type)

    """For each seed and temperature, generate a sketch and save the figure"""

    for i, temp in enumerate(temperatures):
        for j, seed in enumerate(seeds):
            name = str(temp) + "_" + str(seed)
            strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim,
                            time_step=max_seq_len, random_state = seed, cond_gen=cond_gen, 
                            device=device, bi_mode= mode, temperature = temp)
            draw_image(strokes,save=True,save_dir='drawings/unconditional/'+data_type+'/', color = color(i),
            name = name)
            
    """Load the saved figures and plot them all in a common frame along with the original sketches"""

    fig, ax = plt.subplots(10, 8, figsize = (30,30))

    for i, temp in enumerate(temperatures):
        for j, seed in enumerate(seeds):
            name = str(temp) + "_" + str(seed)            
            im = plt.imread('drawings/unconditional/'+data_type+'/'+name+'.png')
            ax[i,j].axis('off')
            ax[i,j].imshow(im)

    plt.axis('off')
    plt.savefig('drawings/unconditional/'+data_type+'/complete.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()