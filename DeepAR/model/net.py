'''Defines the neural network, loss function and metrics'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

logger = logging.getLogger('DeepAR.Net')

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)
        
        # Main LSTM layer
        self.lstm = nn.LSTM(input_size=1+params.cov_dim+params.embedding_dim,
                          hidden_size=params.lstm_hidden_dim,
                          num_layers=params.lstm_layers,
                          bias=True,
                          batch_first=False,
                          dropout=params.lstm_dropout)
        
        # add residual connection 
        self.skip_connection = nn.Linear(1+params.cov_dim+params.embedding_dim, 
                                       params.lstm_hidden_dim)
        
        # Output layers
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, idx, hidden, cell):
        # 1. Embedding processing
        onehot_embed = self.embedding(idx)
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        
        # 2. Save input for residual connection
        skip_connection = self.skip_connection(lstm_input)
        
        # 3. LSTM processing
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # 4. Residual connection
        final_hidden = hidden + skip_connection[-1].unsqueeze(0).repeat(hidden.size(0), 1, 1)

        # 5. Calculate output
        hidden_permute = final_hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, 
                         device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, 
                         device=self.params.device)

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                       device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                         id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred
    
            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma
    
        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma

# new loss: good capture at small fluctuation, but need improve accuracy
def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''Enhanced loss function focusing on matching exact fluctuation amplitudes'''
    zero_index = (labels != 0)
    mu_valid = mu[zero_index]
    labels_valid = labels[zero_index]
    
    if len(mu_valid) < 2:
        return torch.tensor(0.0, device=mu.device)
    
    # 1. Base likelihood loss (降低权重)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood_loss = -torch.mean(distribution.log_prob(labels[zero_index]))
    
    # 2. 波动幅度匹配损失
    pred_diff = mu_valid[1:] - mu_valid[:-1]
    true_diff = labels_valid[1:] - labels_valid[:-1]
    
    # 直接惩罚波动幅度差异，不使用权重
    amplitude_loss = torch.mean(torch.abs(torch.abs(pred_diff) - torch.abs(true_diff)))
    
    # 3. 波动方向损失
    direction_match = torch.sign(pred_diff) * torch.sign(true_diff)
    direction_loss = torch.mean(F.relu(1.0 - direction_match))  # 当方向不匹配时损失为2
    
    # 4. 波动频率损失
    pred_changes = torch.where(torch.abs(pred_diff) > 1e-5, 1.0, 0.0)
    true_changes = torch.where(torch.abs(true_diff) > 1e-5, 1.0, 0.0)
    frequency_loss = F.mse_loss(pred_changes, true_changes)
    
    # 5. 相对波动幅度损失
    relative_amplitude = torch.abs(pred_diff) / (torch.abs(true_diff) + 1e-6)
    relative_loss = torch.mean(torch.abs(relative_amplitude - 1.0))
    
    # 组合损失，强调波动幅度匹配
    total_loss = (0.2 * likelihood_loss +     # 降低基础损失权重
                 3.0 * amplitude_loss +        # 增加波动幅度匹配权重
                 2.0 * direction_loss +        # 保持方向匹配
                 1.0 * frequency_loss +        # 确保捕捉到每个波动
                 2.0 * relative_loss)          # 确保波动幅度比例正确
    return total_loss

# new ND
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    # Modified ND (Normalized Deviation) calculation for stock returns
    zero_index = (labels != 0)
    
    # Convert to returns/percentage changes for comparison
    mu_returns = (mu[1:] - mu[:-1]) / (mu[:-1] + 1e-6)
    label_returns = (labels[1:] - labels[:-1]) / (labels[:-1] + 1e-6)
    
    if relative:
        diff = torch.mean(torch.abs(mu_returns[zero_index[1:]] - label_returns[zero_index[1:]])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu_returns[zero_index[1:]] - label_returns[zero_index[1:]])).item()
        summation = torch.sum(torch.abs(label_returns[zero_index[1:]])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result