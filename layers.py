import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops

from utils import normal

class ChebConv_Coma(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        # self.weight[0].shape:torch.Size([3, 16])
        # self.weight[1].shape:torch.Size([3, 16])
        out = torch.matmul(Tx_0, self.weight[0])
        # out.shape:torch.Size([16, 5023, 16])
        x = x.transpose(0,1)
        Tx_0 = x
        if self.weight.size(0) > 1:
            #x.shape:torch.Size([5023, 16, 3])
            # edge_index.shape: [2, 29990 ]
            # norm.shape: [29990]
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            #Tx_1.shape: torch.Size([5023, 16, 3])

            Tx_1_transpose = Tx_1.transpose(0, 1)
            #matmul: torch.Size([16, 5023, 16])

            out = out + torch.matmul(Tx_1_transpose, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_2_transpose = Tx_2.transpose(0, 1)
            out = out + torch.matmul(Tx_2_transpose, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out.transpose(0,1)

    def message(self, x_j, norm):
        #x_j.shape:torch.Size([5023, 29990, 3])
        # norm.shape:torch.Size([29990])
        #norm.view(-1,1,1):torch.Size([29990, 1, 1])
        # I think the correct code is the one commented below.
        #norm.view(1,-1,1).shape:torch.Size([1, 29990, 1])
        return norm.view(1,-1,1) * x_j


class Pool(MessagePassing):
    def __init__(self):
        super(Pool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat,  dtype=None):
        #x = x.transpose(0,1)
        # forward pool x:shape torch.Size([16, 5023, 16])
        pool_mat2 = pool_mat
        pool_mat2 = pool_mat2.transpose(0,1)
        print(x.shape)
        print('mat2.shape', pool_mat2.shape)
        print('mat2.size',pool_mat2.size())
        # x.shape: [ 5023, 16, 16]
        # edge_index --> pool_mat._indices().shape : [ 2, 1256 ]
        # norm --> pool_mat._values().shape) : [ 1256 ]
        # pool_mat.size():[ 1256, 5023 ]
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat2.size())

        return out.transpose(0,1)

    def message(self, x_j, norm):
        # x_j.shape: [5023, 1256, 16]
        # norm.shape: [1256]
        #return norm.view(-1, 1, 1) * x_j
        return norm.view(1, -1, 1) * x_j


