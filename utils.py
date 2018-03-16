import torch
import numpy as np

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)

def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)

def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)