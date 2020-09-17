import torch
import torch.nn as nn
import numpy as np

SCALE_LOGSIGMA_MIN_MAX = (-20,2)

str_to_activation = {
    'elu': nn.ELU(),
    'hardshrink': nn.Hardshrink(),
    'hardtanh': nn.Hardtanh(),
    'leakyrelu': nn.LeakyReLU(),
    'logsigmoid': nn.LogSigmoid(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'rrelu': nn.RReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'logsoftmax': nn.LogSoftmax(),
    'softshrink': nn.Softshrink(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink(),
    'softmin': nn.Softmin(),
    'softmax': nn.Softmax(dim=1),
    'none': None
    }

def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    return initialize

class BaseNetwork(torch.nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Actor(BaseNetwork):

    def __init__(self, input_dim, output_dim, n_layers, layer_sizes, hidden_activation = "relu",
                output_activation = None, bias = True):
        super(Actor, self).__init__()
        assert n_layers == len(layer_sizes), "length of layer_sizes should matches n_layers"
        model = []
        prev_dim = input_dim
        
        for h in layer_sizes:
            model.append(nn.Linear(prev_dim, h, bias = bias))
            model.append(str_to_activation[hidden_activation])
            prev_dim = h
        
        # output layer
        model.append(nn.Linear(prev_dim, output_dim*2, bias = bias))
        self.model = nn.Sequential(*model).apply(initialize_weights(torch.nn.init.kaiming_uniform_))

    def forward(self, states):

        mu, log_sigma = torch.chunk(self.model(states.float()), 2, dim = -1)
        log_sigma = torch.clamp(log_sigma, min = SCALE_LOGSIGMA_MIN_MAX[0], max = SCALE_LOGSIGMA_MIN_MAX[1])

        sigma = log_sigma.exp()

        distr = torch.distributions.Normal(mu,sigma)

        raw_actions = distr.rsample()
        clipped_actions = torch.tanh(raw_actions)

        # log probs 
        log_probs = distr.log_prob(raw_actions)
        # log_probs -= torch.log(1 - clipped_actions.pow(2) + 1e-6)
        log_probs -= (2*(np.log(2) - raw_actions - nn.functional.softplus(-2*raw_actions)))
        # log_probs = log_probs.sum(1, keepdim = True)
        #clipped mu
        clipped_mu = torch.tanh(mu)

        return clipped_actions, clipped_mu, log_probs, distr

class LyapunovCritic(BaseNetwork):
    def __init__(self, state_dim, action_dim, output_dim, n_layers, layer_sizes, hidden_activation = "relu",
                output_activation = None, bias = True):
        super(LyapunovCritic, self).__init__()
        assert n_layers == len(layer_sizes), "length of layer_sizes should matches n_layers"
        model = []
        self.ll1 = nn.Linear(state_dim, layer_sizes[0], bias = False)
        self.ll2 = nn.Linear(action_dim, layer_sizes[0], bias = False)
        self.bias = torch.autograd.Variable(torch.zeros(1, layer_sizes[0]), requires_grad = True)
        self.ll1.apply(initialize_weights(nn.init.kaiming_uniform_))
        self.ll2.apply(initialize_weights(nn.init.kaiming_uniform_))
        prev_dim = layer_sizes[0]
        self.hidden_activation = hidden_activation
        for h in layer_sizes[1:]:
            model.append(nn.Linear(prev_dim, h, bias = bias))
            model.append(str_to_activation[hidden_activation])
            prev_dim = h
        # model.append(nn.Linear(prev_dim, 1, bias = bias))

        self.model = nn.Sequential(*model).apply(initialize_weights(nn.init.kaiming_uniform_))
    
    def forward(self, states, actions):
        s = self.ll1(states.float())
        a = self.ll2(actions.float())
        x = str_to_activation[self.hidden_activation](s + a + self.bias)
        x = self.model(x)
        # output = x**2
        output = torch.unsqueeze(torch.sum(x**2, dim=1), dim=1)

        return output

def test_model_py():
    actor = Actor(10, 2, 3, [32,16,4])
    critic = LyapunovCritic(10, 2, 2, 3, [32,16,4])


if __name__ == "__main__":
    test_model_py()