from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond
import torch
data_type = 'cat' # can be kanji character or cat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type)
strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, random_state= 98,
                                               cond_gen=cond_gen, device=device, bi_mode= mode)
draw_image(strokes,save=True,save_dir='drawings/')