import numpy as np
import torch
import matplotlib.pyplot as plt

from data_load import get_data
from eval_skrnn import  load_pretrained_congen, plot_dataset, draw_image
from matplotlib import cm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bi_mode = 2 
data_type = 'interpolation'
temperature = 0.1
interpolation_mode = 'slerp'  #'slerp' or 'linear'

n_alpha = 10
alpha_list = np.linspace(0,1,n_alpha)
color =  cm.get_cmap('brg', n_alpha*2)
    
data_type_1 = "cat"
data_type_2 = "cake"

im_nbr_1 = 1000
im_nbr_2 = 1000

"""Redefine subfunctions of function skrnn_sample in model.py"""

def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf
    
def get_pi_id(x, dist, temp=1.0):            
# implementing the cumulative index retrieval
    dist = adjust_temp(np.copy(dist.detach().cpu().numpy()), temp)
    N = dist.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += dist[i]
        if (accumulate >= x):
            return i
    return -1
    
def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0):
    s1 *= temp * temp
    s2 *= temp * temp
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


"""Interpolation function: two sorts of interpolation : linear or spherical-linear"""


def interpolation(sketch_1, sketch_2, alpha, temperature, interp_mode = 'linear'):
    
    hidden_enc = (torch.zeros(bi_mode, 1, hid_dim, device=device), torch.zeros(bi_mode, 1, hid_dim, device=device))
    hidden_dec = (torch.zeros(1, 1, hid_dim, device=device), torch.zeros(1, 1, hid_dim, device=device))
  
    """Compute latent variables of each sketch"""
    z_1, hidden_dec_1, mu_1, sigma_1 = encoder(sketch_1, hidden_enc)
    z_2, hidden_dec_2, mu_2, sigma_2 = encoder(sketch_2, hidden_enc)
    
    """Compute latent variables of the interpolated sketch"""
    if interp_mode == 'linear':
        z = alpha * z_1 + (1.0 - alpha) * z_2 
        hidden_dec = (alpha * hidden_dec_1[0] + (1.0 - alpha) * hidden_dec_2[0], 
                alpha * hidden_dec_1[1] + (1.0 - alpha) * hidden_dec_2[1])
    else :
        cos_theta_z = torch.vdot(z_1.squeeze(0), z_2.squeeze(0))/(torch.norm(z_1.squeeze(0))*torch.norm(z_2.squeeze(0)))
        theta_z = torch.acos(cos_theta_z)
        z = z_1 * torch.sin(alpha*theta_z)/torch.sin(theta_z) + z_2 * torch.sin((1.0-alpha)*theta_z)/torch.sin(theta_z)

        cos_theta_h = torch.vdot(torch.cat([hidden_dec_1[0], hidden_dec_1[1]],2).squeeze(0).squeeze(0),
                              torch.cat([hidden_dec_2[0], hidden_dec_2[1]],2).squeeze(0).squeeze(0))/(torch.norm(torch.cat([hidden_dec_1[0], hidden_dec_1[1]],2).squeeze(0).squeeze(0))*torch.norm(torch.cat([hidden_dec_2[0], hidden_dec_2[1]],2).squeeze(0).squeeze(0)))
        theta_h = torch.acos(cos_theta_h)
        hidden_dec = (hidden_dec_1[0] * torch.sin(alpha*theta_h)/torch.sin(theta_h) + hidden_dec_2[0] * torch.sin((1.0-alpha)*theta_h)/torch.sin(theta_h),
                     hidden_dec_1[1] * torch.sin(alpha*theta_h)/torch.sin(theta_h) + hidden_dec_2[1] * torch.sin((1.0-alpha)*theta_h)/torch.sin(theta_h))
    time_step = max_seq_len
    end_stroke = time_step
    
    """Compute the decoded interpolated sketch (as done in skrnn_sample in model.py)"""
    start=[0,0,1,0,0]
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    strokes = np.zeros((time_step, 5), dtype=np.float32)
    mixture_params = []

    
    for i in range(time_step):
        gmm_params, hidden_dec = decoder(prev_x.unsqueeze(0).unsqueeze(0), z, hidden_dec)
        q, pi, mu1, mu2, s1, s2, rho = gmm_params[0][0],gmm_params[1][0],gmm_params[2][0],gmm_params[3][0],gmm_params[4][0],gmm_params[5][0],gmm_params[6][0]
        
        idx = get_pi_id(np.random.random(), pi, temperature)        
        eos_id = get_pi_id(np.random.random(), q, temperature)
        
        eos = [0, 0, 0]
        eos[eos_id] = 1
    
        next_x1, next_x2 = sample_gaussian_2d(mu1[idx].detach().cpu().numpy(), mu2[idx].detach().cpu().numpy(), 
                            s1[idx].detach().cpu().numpy(), s2[idx].detach().cpu().numpy(), 
                            rho[idx].detach().cpu().numpy())
        
        mixture_params.append([float(mu1[idx].detach().cpu()),float(mu2[idx].detach().cpu()), float(s1[idx].detach().cpu()), 
                            float(s2[idx].detach().cpu()), float(rho[idx].detach().cpu()), q])
        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
        if eos[-1] == 1:
            end_stroke = i+1
            break
        prev_x[0], prev_x[1], prev_x[2], prev_x[3], prev_x[4] = next_x1, next_x2, eos[0], eos[1], eos[2]
        
    mix_params = np.array(mixture_params)
    
    return strokes[:end_stroke,[0,1,3]], mix_params
    

if __name__ == "__main__":

    data_enc_1, data_dec_1, max_seq_len_1 = get_data(data_type = data_type_1, max_len=200)
    data_enc_2, data_dec_2, max_seq_len_2 = get_data(data_type = data_type_2, max_len=200)
    max_seq_len = max(max_seq_len_1, max_seq_len_2)
    
    plot_dataset(data_enc_1, im_nbr_1)
    plot_dataset(data_enc_2, im_nbr_2)
    
    sk_1 = torch.tensor(data_enc_1[im_nbr_1], dtype=torch.float, device=device).unsqueeze(0)
    sk_2 = torch.tensor(data_enc_2[im_nbr_2], dtype=torch.float, device=device).unsqueeze(0)
    
    encoder, decoder, hid_dim, latent_dim, cond_gen, mode, device = load_pretrained_congen('interpolation')
        
    """For each alpha, compute the interpolated sketch, draw it and save the plot"""
    for i, alpha in enumerate(alpha_list):
        
        name = "interpolate_" + interpolation_mode + '_'+str(round(alpha,2))     
        strokes, mix_params = interpolation(sk_1, sk_2, alpha, temperature, interp_mode= interpolation_mode)
        draw_image(strokes,save=True,save_dir='drawings/conditional/'+data_type+'/', color = color(i),
            name = name)

    """Plot the original non-interpolated sketches and save the plots"""
    name_1 = "original_"+data_type_1+"_"+'_'+str(im_nbr_1)
    name_2 = "original_"+data_type_2+"_"+'_'+str(im_nbr_2)
    
    plot_dataset(data_enc_1, im_nbr_1, save = True, save_dir = 'drawings/conditional/'+data_type+'/',
                  name = name_1, color='green')
    plot_dataset(data_enc_2, im_nbr_2, save = True, save_dir = 'drawings/conditional/'+data_type+'/',
                 name = name_2, color='green')
    
    """Get back the saved plots and put them all in a common frame"""
    fig, ax = plt.subplots(1, n_alpha+2, figsize = (35,12))
    
    im = plt.imread('drawings/conditional/'+data_type+'/'+name_2+'.png')
    ax[0].axis('off')
    ax[0].imshow(im)
    
    im = plt.imread('drawings/conditional/'+data_type+'/'+name_1+'.png')
    ax[n_alpha+1].axis('off')
    ax[n_alpha+1].imshow(im)        
    
    for i, alpha in enumerate(alpha_list):
            
        name = "interpolate_" +interpolation_mode+'_'+ str(round(alpha,2))            
        im = plt.imread('drawings/conditional/'+data_type+'/'+name+'.png')
        ax[i+1].axis('off')
        ax[i+1].imshow(im)

    plt.axis('off')
    plt.savefig('drawings/conditional/'+data_type+'/'+interpolation_mode+'_complete.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()
    