import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
from common import GaussianSmearing, SPDMLP, MLP, batch_hybrid_edge_connection, NONLINEARITIES

class Feature_Joint(nn.Module):
    def __init__(self, ):
        super(Feature_Joint, self).__init__()

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        return joint_feature

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, act_fn="relu", update_x=True, norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.edge_mlp = SPDMLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2)
        #MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn='relu', act_last=True)
        self.FJ1 = Feature_Joint()
        self.FJ2 = Feature_Joint()
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = SPDMLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2)

    def forward(self, h, x, edge_index):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        mij = self.edge_mlp(self.FJ1(hi, hj))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])
        h = h + self.node_mlp(self.FJ2(mi, h))
        return h, x


class SPD_GNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, k=16, update_x=True, norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.norm = norm
        self.k = k
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(16, 16,
                                update_x=self.update_x, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, prot_batchs):
        edge_index = knn_graph(x, k=self.k, batch=prot_batchs, flow='source_to_target')
        return edge_index

    def forward(self, h, x, prot_batchs, return_all=False):
        all_x = [x]
        all_h = [h]
        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, prot_batchs)
            h, x = layer(h, x, edge_index)
            all_x.append(x)
            all_h.append(h)
        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return h, x, outputs
