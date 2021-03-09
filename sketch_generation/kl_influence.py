from data_load import get_data
from model import skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond, load_pretrained_congen, plot_dataset
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

data_type = 'cat' # can be kanji character or cat or cake
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cond_gen = True


seeds = np.random.randint(0, 2500, 4)
weights = [0.25, 0.5, 1.0]
color =  cm.get_cmap('brg', len(weights)*2)

"""Conditional generation"""
if cond_gen == True:
    
    data_enc, data_dec, max_seq_len = get_data(data_type = data_type, max_len=200)
    
    """For each seed and weight, generate a sketch and save the figure"""
    for i, weight in enumerate(weights):
        encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_congen(data_type, weight)
       
        for j, seed in enumerate(seeds):
            inp_enc = torch.tensor(data_enc[seed], dtype=torch.float, device=device).unsqueeze(0)
            name = "original_"+str(seed)
            plot_dataset(data_enc, seed, save = True, save_dir = 'drawings/conditional/'+data_type+'/',
                         name = 'original_'+str(seed), color='green')
            name = str(weight) + "_" + str(seed)     
            strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim,
                            time_step= max_seq_len, random_state = seed, cond_gen=True, 
                            device=device, bi_mode= mode, temperature = 1.0, inp_enc = inp_enc)
            draw_image(strokes,save=True,save_dir='drawings/conditional/'+data_type+'/', color = color(i),
            name = name)

    """Load the saved figures and plot them all in a common frame along with the original sketches"""
    fig, ax = plt.subplots(4, 4, figsize = (30,30))

    for j, seed in enumerate(seeds):
        name = "original_" + str(seed)            
        im = plt.imread('drawings/conditional/'+data_type+'/'+name+'.png')
        
        ax[0, j].axis('off')
        ax[0, j].imshow(im)
        
        for i, weight in enumerate(weights):
            
            name = str(weight) + "_" + str(seed)            
            im = plt.imread('drawings/conditional/'+data_type+'/'+name+'.png')
            ax[i+1,j].axis('off')
            ax[i+1,j].imshow(im)

    plt.axis('off')
    plt.savefig('drawings/conditional/'+data_type+'/complete.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()