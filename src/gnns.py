import torch
import torch_geometric

class Swish(torch.nn.Module):
    def __init__(self,inplace=True):
        super(Swish,self).__init__()
        self.inplace=inplace
    def forward(self,x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)

class GIN(torch.nn.Module):
    def __init__(self, input_size, model_size, nlayer, device):
        super().__init__()
        self.model_size = model_size
        self.gnns = [torch_geometric.nn.GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_size, model_size),
            Swish())).to(device)]
        for _ in range(1, nlayer):
            self.gnns.append(torch_geometric.nn.GINConv(torch.nn.Sequential(
                torch.nn.Linear(model_size, model_size),
                Swish())).to(device))
    def forward(self, x, edge_index):
        output = self.gnns[0](x, edge_index)
        for i, gnn in enumerate(self.gnns):
            if i == 0: continue
            output =  gnn(output, edge_index)
        return output