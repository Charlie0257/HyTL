import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor

# All tensor will be created as float32 as default
DEFAULT_TENSOR_TYPE = th.float32

# Use CPU as default. Small network on CPU is much faster than on GPU.
DEFAULT_DEVICE = "cpu"  # th.device("cuda") if th.cuda.is_available() else th.device("cpu")

def default_tensor(inpt) -> th.Tensor:
    return th.tensor(inpt,
                     dtype=DEFAULT_TENSOR_TYPE,
                     device=DEFAULT_DEVICE)

def as_numpy(inpt) -> np.ndarray:
    if isinstance(inpt, Tensor):
        return inpt.detach().cpu().numpy()
    else:
        return np.array(inpt)

class GRUEulerNeuralOde(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 der_bound: float = 10.0):
        super(GRUEulerNeuralOde, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, input_size)
        self.tanh = nn.Tanh()
        self.der_bound = der_bound

    def forward(self, x: th.Tensor, hn: th.Tensor):
        if hn is not None:
            hn = hn.detach()

        logist, hn = self.gru(x, hn)
        der = self.linear_layer(logist)
        der = self.tanh(der)

        return x + der * self.der_bound, logist, hn

class GruOdePolicy(th.nn.Module):
    def __init__(self,
                 input_size: int,
                 path_max_len: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 der_bound: float = 10.0):
        super(GruOdePolicy, self).__init__()

        self.path_max_len = path_max_len

        self.gru_ode = GRUEulerNeuralOde(input_size, hidden_size, num_layers, der_bound)
        self.var_layer = th.nn.Linear(hidden_size, input_size)
        self.sample_dist_cls = th.distributions.Normal

        self._eps = default_tensor(1e-1)

    def forward(self, initial_state: th.Tensor, deterministic: bool = False) -> th.Tensor:
        hn = None

        if len(initial_state.shape) <= 2:
            initial_state = initial_state.view(-1, 1, initial_state.shape[-1])

        path = [initial_state]
        state = initial_state.detach()

        for _ in range(self.path_max_len - 1):
            state_mean, logist, hn = self.gru_ode(state, hn)

            if deterministic:
                state = state_mean
            else:
                scale = th.abs(self.var_layer(logist)) + self._eps
                dist = self.sample_dist_cls(state_mean, scale)
                state = dist.rsample()

            path.append(state)
            state = state.detach()

        path = th.cat(path, dim=1)

        return path

    def predict(self,
                initial_state: np.ndarray,
                deterministic: bool = False) -> np.ndarray:
        return as_numpy(self.forward(default_tensor(initial_state),
                                     deterministic=deterministic).squeeze())

    def log_prob(self, path_tensor: th.Tensor):
        if len(path_tensor.shape) == 2:
            path_tensor = path_tensor.view(-1, *path_tensor.shape)

        means, logist, hns = self.gru_ode(path_tensor, None)
        scales = th.abs(self.var_layer(logist)) + self._eps

        dist = self.sample_dist_cls(means, scales)
        log_probs = dist.log_prob(path_tensor).clip(-0.1, th.inf)

        return log_probs.mean(axis=(1, 2))
