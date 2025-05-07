import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_layers(in_channels, hidden_channels, num_layers, batch_norm=False, layer_norm=False):
    layers = []

    for i in range(num_layers):
        mlp_layer = nn.Linear(in_channels, hidden_channels)
        if batch_norm:
            layers += [mlp_layer, nn.BatchNorm1d(hidden_channels)]
        elif layer_norm:
            layers += [mlp_layer, nn.LayerNorm(hidden_channels)]
        else:
            layers += [mlp_layer]
        in_channels = hidden_channels
    # proj = nn.Linear(hidden_channels, hidden_channels)
    # layers += [proj]
    return torch.nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, batch_norm=False, layer_norm=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.layer2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.ln = nn.LayerNorm(hidden_channels)

        self.use_bn = batch_norm
        self.use_ln = layer_norm
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_ln:
            x = self.ln(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(Predictor, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        h = self.linears[self.num_layers - 1](h)
        return h


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class IGCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, proj_layer, tau, beta, lamda,
                 batch_norm=False, layer_norm=False):
        super(IGCL, self).__init__()
        self.tau = tau
        self.lamda = lamda
        self.encoder = mlp_layers(in_channels, hidden_channels, num_layers, batch_norm, layer_norm)

        self.encoder_target = copy.deepcopy(self.encoder)
        self.projector = Predictor(hidden_channels, hidden_channels, proj_layer)
        self.ema = EMA(beta)
        set_requires_grad(self.encoder_target, False)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def pos_score(self, h, v, P):
        q = F.normalize(self.projector(h))
        v = F.normalize(v)
        par_indices = P.coalesce().indices()[1]
        neg_indices = torch.randint(0, h.size(0), (v.size(0),))
        pos_sim = (torch.sum(v*q[par_indices], dim=1) / self.tau).unsqueeze(1)
        neg_sim = (torch.sum(v*q[neg_indices], dim=1) / self.tau).unsqueeze(1)

        return torch.exp(torch.spmm(P.T, pos_sim)), torch.exp(torch.spmm(P.T, neg_sim))

    def neg_score(self, h):
        h = F.normalize(h)
        neg_sim = torch.exp(torch.mm(h, h.t()) / self.tau)
        return neg_sim

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.encoder_target is not None, 'target encoder has not been created yet'
        for current_params, ma_params in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.ema.update_average(old_weight, up_weight)

    def forward(self, x, P, coarse_g):
        h = self.encoder(x)
        h = torch.spmm(P.T, h)
        x = self.encoder_target(x)
        pos_score, neg_par = self.pos_score(h, x, P)
        neg_score = self.neg_score(h)
        partition_score = torch.sparse.mm(coarse_g, neg_score)
        loss = (-torch.log(pos_score + self.lamda*partition_score) + (torch.log(pos_score + neg_score.sum(1)))).mean()
        #loss = (-torch.log(pos_score + self.lamda*partition_score) + (torch.log(pos_score + neg_par + neg_score.sum(1)))).mean()
        return loss

    def get_emb(self, x, g, k):
        x = self.encoder(x)
        for _ in range(k):
            x = torch.spmm(g, x)
        return torch.nn.functional.relu(x)
