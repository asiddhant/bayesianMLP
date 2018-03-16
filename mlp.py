import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Linear(input_size, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, output_size)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, input_var):
        
        hidden1 = F.relu(self.layer1(input_var))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.layer2(hidden1))
        hidden2 = self.dropout(hidden2)
        output = self.layer3(hidden2)
        return output
    
class BayesLinear(nn.Module):
    def __init__(self, input_size, output_size, sigma_prior):
        super(BayesLinear, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_prior = sigma_prior
        
        self.W_mu = nn.Parameter(torch.Tensor(input_size, output_size).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(input_size, output_size).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(output_size).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(output_size).uniform_(-0.01, 0.01))
        
        self.lpw = 0
        self.lqw = 0
        
    def forward(self, input_var, infer=False):
        
        batch_size = input_var.size()[0]
        if infer:
            output = torch.mm(input_var, self.W_mu) + self.b_mu.expand(batch_size, self.output_size)
            return output
        
        epsilon_W = Variable(torch.Tensor(self.input_size, self.output_size).normal_(0, self.sigma_prior)).cuda()
        epsilon_b = Variable(torch.Tensor(self.output_size).normal_(0, self.sigma_prior)).cuda()
        
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(input_var, W) + b.expand(batch_size, self.output_size)
        
        self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
        self.lqw = (log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + 
                   log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum())
        return output
    
class BayesMLP(nn.Module):
    def __init__(self, input_size, output_size, sigma_prior):
        super(BayesMLP, self).__init__()
        
        self.layer1 = BayesLinear(input_size, 200, sigma_prior)
        self.layer2 = BayesLinear(200, 200, sigma_prior)
        self.layer3 = BayesLinear(200, output_size, sigma_prior)

    def forward(self, input_var, infer=False):
        
        hidden1 = F.relu(self.layer1(input_var, infer))
        hidden2 = F.relu(self.layer2(hidden1, infer))
        output = self.layer3(hidden2, infer)
        return output

    def get_lpw_lqw(self):
        
        lpw = self.layer1.lpw + self.layer2.lpw + self.layer3.lpw
        lqw = self.layer1.lqw + self.layer2.lqw + self.layer3.lqw
        return lpw, lqw
        