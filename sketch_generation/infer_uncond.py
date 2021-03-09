from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond
import torch
import numpy as np
data_type = 'cake' # can be kanji character or cat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_uncond(data_type)

t_step = 135
strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, random_state=10,
                                               cond_gen=cond_gen, device=device, bi_mode= mode, temperature = 0.2)

draw_image(strokes,save=False)
