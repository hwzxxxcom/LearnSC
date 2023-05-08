import torch
import torch_geometric
from gnns import GIN

class Interactor(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.query_gnn = GIN(input_size, output_size, 1, device)
        self.graph_gnn = GIN(input_size, output_size, 1, device)
        self.dummy_gnn = torch_geometric.nn.GraphSAGE(output_size, output_size, 1).to(device)
    def forward(self, xg: torch.Tensor, eg: torch.Tensor, n_orig_gnode: int, match:dict[int,int], 
                      xq: torch.Tensor, eq: torch.Tensor, n_orig_qnode: int):
        outq = xq# self.query_gnn(xq, eq)
        outg = xg #self.graph_gnn(xg, eg)
        # Construct Dummy Node Graph
        xd = torch.cat((outq[n_orig_qnode:], outg[n_orig_gnode:]), dim = 0)
        # ed
        vg = torch.tensor(list(match.keys()), dtype=torch.long) - n_orig_gnode + xq.shape[0] - n_orig_qnode
        uq = torch.tensor(list(match.values()), dtype=torch.long) - n_orig_qnode
        ed = torch.stack([uq, vg], dim = 0)
        outd = self.dummy_gnn(xd, ed)
        maskq = torch.cat([torch.ones([n_orig_qnode, 1]), torch.zeros([xq.shape[0] - n_orig_qnode, 1])], dim = 0)
        maskg = torch.cat([torch.ones([n_orig_gnode, 1]), torch.zeros([xg.shape[0] - n_orig_gnode, 1])], dim = 0)
        Pq = torch.sparse.FloatTensor(
            torch.stack([torch.arange(n_orig_qnode, xq.shape[0]), torch.arange(xq.shape[0] - n_orig_qnode)], dim = 0), 
            torch.ones(xq.shape[0] - n_orig_qnode), torch.Size([xq.shape[0], xd.shape[0]]))
        Pg = torch.sparse.FloatTensor(
            torch.stack([torch.arange(n_orig_gnode, xg.shape[0]), torch.arange(xg.shape[0] - n_orig_gnode) + xq.shape[0] - n_orig_qnode], dim = 0), 
            torch.ones(xg.shape[0] - n_orig_gnode), torch.Size([xg.shape[0], xd.shape[0]]))
        outq = outq * maskq + Pq.matmul(outd)
        outg = outg * maskg + Pg.matmul(outd)
        return outg, outq
        