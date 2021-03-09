"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_load import get_data
from model import encoder_skrnn, decoder_skrnn

def draw_image(a, color=None, save=False, save_dir=None, name=None):
    x = np.cumsum(a[:,0], 0)
    y = np.cumsum(a[:,1], 0)
    stroke = np.stack([x,y],1)
    cuts = np.where(a[:,2]>0)[0] + 1
    
    strokes = np.split(stroke[:-1],cuts)
    
    fig = plt.figure(figsize = (12,12))
    ax1 = fig.add_subplot(111)
    
    for s in strokes:
        plt.plot(s[:,0],-s[:,1],color=color)
    if save:
        if name==None:
            file_name = save_dir+str(np.random.randint(0,5000))
            print(file_name)
        else:
            file_name = save_dir + name + ".png"
            print(file_name)
        plt.axis('off')
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.show()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_uncond(data_type):
    ####################################################
    # default parameters, do not change
    hidden_enc_dim = 256
    hidden_dec_dim = 256
    n_layers = 1
    num_gaussian = 20 
    dropout_p = 0.2
    batch_size = 50
    latent_dim = 64 

    rnn_dir = 2 # 1 for unidirection,  2 for bi-direction
    bi_mode = 2 # bidirectional mode:- 1 for addition 2 for concatenation
    cond_gen = False

    if not cond_gen:
        rnn_dir = 1
        bi_mode = 1

    ####################################################
    
    encoder = encoder_skrnn(input_size = 5, hidden_size = hidden_enc_dim, hidden_dec_size=hidden_dec_dim,\
                        dropout_p = dropout_p,n_layers = n_layers, batch_size = batch_size, latent_dim = latent_dim,\
                        device = device, cond_gen= cond_gen, bi_mode= bi_mode, rnn_dir = rnn_dir).to(device)
    
    decoder = decoder_skrnn(input_size = 5, hidden_size = hidden_dec_dim, num_gaussian = num_gaussian,\
                            dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                            latent_dim = latent_dim, device = device, cond_gen= cond_gen).to(device)

    if data_type == 'cat':
        encoder.load_state_dict(torch.load('saved_model/UncondEnc_cat.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/UncondDec_cat.pt',map_location='cuda:0')['model'])
    
    elif data_type =="cake":
        encoder.load_state_dict(torch.load('saved_model/UncondEnc_cake.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/UncondDec_cake.pt',map_location='cuda:0')['model'])    
    
    else:
        encoder.load_state_dict(torch.load('saved_model/UncondEnc_kanji.pt', map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/UncondDec_kanji.pt', map_location='cuda:0')['model'])

    return encoder, decoder, hidden_enc_dim, latent_dim, cond_gen, bi_mode, device

def load_pretrained_congen(data_type, wkl=1.0):
    ####################################################
    # default parameters, do not change
    hidden_enc_dim = 256
    hidden_dec_dim = 256
    n_layers = 1
    num_gaussian = 20 
    dropout_p = 0.2
    batch_size = 50
    latent_dim = 64 

    rnn_dir = 2 # 1 for unidirection,  2 for bi-direction
    bi_mode = 2 # bidirectional mode:- 1 for addition 2 for concatenation
    cond_gen = True

    if not cond_gen:
        rnn_dir = 1
        bi_mode = 1

    ####################################################
    
    encoder = encoder_skrnn(input_size = 5, hidden_size = hidden_enc_dim, hidden_dec_size=hidden_dec_dim,\
                        dropout_p = dropout_p,n_layers = n_layers, batch_size = batch_size, latent_dim = latent_dim,\
                        device = device, cond_gen= cond_gen, bi_mode= bi_mode, rnn_dir = rnn_dir).to(device)
    
    decoder = decoder_skrnn(input_size = 5, hidden_size = hidden_dec_dim, num_gaussian = num_gaussian,\
                            dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                            latent_dim = latent_dim, device = device, cond_gen= cond_gen).to(device)
    if data_type == 'cat':
        encoder.load_state_dict(torch.load('saved_model/condEnc_cat_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/condDec_cat_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])

    elif data_type == 'kanji':
        encoder.load_state_dict(torch.load('saved_model/condEnc_kanji_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/condDec_kanji_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])

    elif data_type == 'cake':
        encoder.load_state_dict(torch.load('saved_model/CondEnc_cake_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/CondDec_cake_w'+str(wkl)+'.pt',map_location='cuda:0')['model'])

    elif data_type == 'interpolation':
        encoder.load_state_dict(torch.load('saved_model/CondEnc_interpolation_w1.0.pt',map_location='cuda:0')['model'])
        decoder.load_state_dict(torch.load('saved_model/CondDec_interpolation_w1.0.pt',map_location='cuda:0')['model'])    
    
    return encoder, decoder, hidden_dec_dim, latent_dim, cond_gen, bi_mode, device

"""Function to plot a sketch from the dataset"""
def plot_dataset(data, im = 10, color = None, save = False, 
                  save_dir = None, name = None):

    stop = np.where(data[im][:,4]==1.0)[0][0] +1
    draw_image(data[im][:stop, [0,1,3]], color = color, save = save,
                      save_dir = save_dir, name = name)

    return