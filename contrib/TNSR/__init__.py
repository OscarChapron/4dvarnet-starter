import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import truncnorm
import contrib.TNSR.TNSR as TNSR
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts


class TNN(pl.LightningModule):
    def __init__(self, rec_weight, opt_fn, ranks, input_shape, sampling_rate = 0.1, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.ranks = ranks
        self.z = nn.Parameter(torch.randn(*ranks))
        self.input_shape = input_shape

        # Initialize V2d and Vd
        self.V2d = nn.ParameterList([
            nn.Parameter(truncnorm(-1, 1).rvs([10, 5])) for _ in range(3)
        ])
        self.Vd = nn.ParameterList([
            nn.Parameter(truncnorm(-1, 1).rvs([input_shape[i], 10])) for i in range(3)
        ])

        self.sampling_rate = sampling_rate
        print(sampling_rate)

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)
    
    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    @staticmethod
    def weighted_mse_mask(err, weight, mask_nan):
        err_valid = err * mask_nan[None, ...]
        # Calculate the number of valid elements
        num_valid = mask_nan.sum()
    
        if num_valid == 0:
            return torch.tensor(1000.0, device=err.device, requires_grad=True)
        weight_valid = weight * mask_nan[None, ...]

        err_w = err_valid * weight_valid[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        err_w_res = err_w.reshape(err_num.size())
        loss = F.mse_loss(err_w_res[err_num], torch.zeros_like(err_w_res[err_num]))
        return loss

    # Total Variation Regularizer
    @staticmethod
    def total_variation(images):
        pixel_dif1 = images[..., 1:] - images[..., :-1]
        pixel_dif2 = images[..., :, 1:] - images[..., :, :-1]
        pixel_dif3 = images[..., :, :, 1:] - images[..., :, :, :-1]
        return torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2)) + torch.sum(torch.abs(pixel_dif3))

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def configure_optimizers(self):
        return self.opt_fn(self)
    
# Define the model architecture
class TNN(nn.Module):
    def __init__(self, input_shape, ranks):
        super(TNN, self).__init__()
        self.input_shape = input_shape
        self.ranks = ranks
        self.z = nn.Parameter(torch.randn(*ranks))
        
        # Initialize V2d and Vd
        self.V2d = nn.ParameterList([
            nn.Parameter(truncnorm(-1, 1).rvs([10, 5])) for _ in range(3)
        ])
        self.Vd = nn.ParameterList([
            nn.Parameter(truncnorm(-1, 1).rvs([input_shape[i], 10])) for i in range(3)
        ])

    def forward(self):
        zd = torch.relu(TNSR.tucker_to_tensor(self.z, self.V2d))
        x_hat = torch.tanh(TNSR.tucker_to_tensor(zd, self.Vd))
        return x_hat

# load the mean
mean = scipy.io.loadmat('data_mean.mat')
data_mean = np.array(mean['data_mean']).astype('float32')

## add noise
noisy_data = ssf + sigma * np.random.randn(N_x,N_y,N_z)

## normalize
ssp_max = np.max(np.abs(noisy_data))
x = noisy_data / ssp_max

## observation
nmod = 3
total = N_x * N_y * N_z
size = [N_x,N_y,N_z]
N = (int)(p*np.prod(size))
index = []
for i in range(N_x):
    for j in range(N_y):
        for k in range(N_z):
            index.append([i,j,k])
index = np.array(index)
ind = index[np.random.choice(total,N,replace=False)]#

ob = np.zeros(size)
for i in range(N):
    ob[tuple(ind[i,:])] = 1

x_ob = x * ob


## TNN model
# Input
I=N_x;J=N_y;K=N_z
input_shape = [I,J,K]
R1=5;R2=5;R3=5
rank = [R1,R2,R3]
z = np.random.randn(R1,R2,R3)
z = tf.Variable(z, dtype=tf.float32)

# decoder
v2d = [stats.truncnorm(-1, 1).rvs([10,5]) for i in range(3)]
V2d = [tf.Variable(v2d[k], dtype=tf.float32) for k in range(3)]
zd = tf.nn.relu(TNSR.tucker_to_tensor(z,V2d))

vd = [stats.truncnorm(-1, 1).rvs([input_shape[i],10]) for i in range(3)]
Vd = [tf.Variable(vd[k], dtype=tf.float32) for k in range(3)]
x_hat = tf.nn.tanh(TNSR.tucker_to_tensor(zd,Vd))


# Training settings
lamb = 0.01  # Regularization parameter
x_ob = torch.randn(*input_shape)  # Observed input data

# Instantiate the model
model = TNN(input_shape=[I, J, K], ranks=[R1, R2, R3])
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training function
def train(model, epochs=100):
    criterion = nn.MSELoss()
    tv_reg = lambda x: lamb * total_variation(x)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat = model()
        loss = criterion(x_ob, x_hat) + tv_reg(x_hat)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Start training
train(model)