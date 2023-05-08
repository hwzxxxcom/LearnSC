import torch
from aggregators import *
from decomposers import *
from gnns import GIN
from utils import *
from interactors import *

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
        
swish = Swish()

class LearnSC(torch.nn.Module):
    def __init__(self, args, input_size, model_size, nlayer = 2):
        super().__init__()
        self.args = args
        self.test = False
        self.model_size = model_size
        self.graph_gnn = GIN(input_size, model_size, nlayer, args.device)
        self.query_gnn1 = GIN(input_size, model_size, 1, args.device)
        self.query_gnn2 = GIN(model_size, model_size, nlayer - 1, args.device)
        self.linear = torch.nn.Linear(model_size, model_size)
        self.linear1 = torch.nn.Linear(model_size, model_size)
        self.linear2 = torch.nn.Linear(model_size, model_size)
        self.linear3 = torch.nn.Linear(model_size, model_size)
        self.linear4 = torch.nn.Linear(model_size, model_size)
        self.interactor = GIN(model_size, model_size, 2, args.device)
        self.interactor1 = Interactor(model_size, model_size, args.device)
        self.aggregate = GIN(2 * model_size, model_size, 2, args.device)
        self.aggregate2 = torch.nn.Linear(2 * model_size, model_size)
        self.estimate = torch.nn.Linear(model_size, 1)
        self.weighter = torch.nn.Linear(2 * model_size , 1)
        self.weighter2 = torch.nn.Linear(model_size + 1, 1)
    def forward(self, xg: torch.Tensor, eg: torch.Tensor, n_orig_gnode: int, match:dict[int,int], 
                      xq: torch.Tensor, eq: torch.Tensor, n_orig_qnode: int, overlap: dict[(int,int),list[int]], subqueries: list[list[int]], itedge, npairs):
        hg, hq = self.graph_gnn(xg, eg), self.query_gnn1(xq, eq)
        transg = torch.zeros((n_orig_gnode, hg.shape[0]))
        transg[:, :n_orig_gnode] = torch.eye(n_orig_gnode)
        transg = transg.detach().to(xg.device)
        orig_hg = transg.mm(hg)
        transq = torch.zeros((n_orig_qnode, hq.shape[0]))
        transq[:, :n_orig_qnode] = torch.eye(n_orig_qnode)
        transq = transq.detach().to(xg.device)
        orig_hq = transq.mm(hq)
        transq1 = torch.zeros((n_orig_qnode + n_orig_gnode, n_orig_qnode)).to(xg.device)
        transq1[:n_orig_qnode, :n_orig_qnode] = torch.eye(n_orig_qnode).to(xg.device)
        transg1 = torch.zeros((n_orig_qnode + n_orig_gnode, n_orig_gnode)).to(xg.device)
        transg1[n_orig_qnode:, :] = torch.eye(n_orig_gnode).to(xg.device)
        itgraph = transq1.mm(orig_hq) + transg1.mm(orig_hg)
        if self.args.no_interaction:
            itgraph = self.interactor(itgraph, torch.LongTensor().reshape((2,0)))
        else:
            itgraph = self.interactor(itgraph, itedge).to(xg.device)
            embedpicker = torch.eye(itgraph.shape[0])
            npair = torch.tensor(npairs, dtype = torch.long).T.to(xg.device) if npairs else torch.LongTensor().reshape(2,0).to(xg.device)
            pos_pair = embedpicker[itedge]
            neg_pair = embedpicker[npair]
            P1 = pos_pair[0].detach()
            P2 = pos_pair[1].detach()
            N1 = neg_pair[0].detach()
            N2 = neg_pair[1].detach()
            X1 = torch.cat((P1, N1), dim = 0).detach().to(xg.device)
            X2 = torch.cat((P2, N2), dim = 0).detach().to(xg.device)
            x1s = X1.mm(itgraph)
            x2s = X2.mm(itgraph)
            ys = torch.cat((torch.ones(pos_pair.shape[1]), -torch.ones(neg_pair.shape[1])), dim = -1).to(xg.device)
        gmsk = torch.tensor([[1]]*n_orig_gnode+[[0]]*(hg.shape[0]-n_orig_gnode),dtype=torch.float).to(xg.device)
        qmsk = torch.tensor([[1]]*n_orig_qnode+[[0]]*(hq.shape[0]-n_orig_qnode),dtype=torch.float).to(xg.device)
        hg = (1-gmsk) * hg + transg.T.mm(transg1.T.mm(itgraph))
        hq = (1-qmsk) * hq + transq.T.mm(transq1.T.mm(itgraph))
        match_count = dict()
        mask, values = torch.ones([hq.shape[0], 1], dtype=torch.float).to(xg.device), torch.zeros(hq.shape, dtype=torch.float).to(xg.device).requires_grad_(False)
        for graph_node in match:
            query_node = match[graph_node]
            if query_node not in match_count: 
                match_count[query_node] = 0
                mask[query_node] *= 0
            match_count[query_node] += 1
        values = values.detach()
        hq = hq * mask.detach()
        hq = hq + values.detach()
        hq = swish(hq + self.query_gnn2(hq, eq))
        readout = torch.stack([torch.sum((self.linear1(hq[subquery])), dim = -2) for subquery in subqueries], dim = 0)
        readout_q = readout
        readout_g = torch.sum(self.linear2(hg[:n_orig_gnode]), dim = -2)
        overlap_feature = torch.zeros(readout.shape, dtype = torch.float).to(xq.device)
        overlap_count = torch.ones((readout.shape[0], 1)).to(xq.device)
        skeleton_edges = []
        for i, j in overlap:
            if i > j: continue
            skeleton_edges.append((i,j))
            overlap_feature[i] += torch.sum(self.linear3(hq[overlap[(i, j)]]), dim = 0)
            overlap_count[i][0] += 1
            overlap_feature[j] += torch.sum(self.linear3(hq[overlap[(i, j)]]), dim = 0)
            overlap_count[j][0] += 1
        overlap_count = overlap_count.detach()
        overlap_feature = overlap_feature / overlap_count
        skeleton_edges = (torch.tensor(skeleton_edges).reshape((2, -1)).type(torch.long)).to(xg.device)# + torch.tensor(n_orig_qnode)).to(xg.device)
        overlap_feature = self.aggregate(torch.cat([readout_q, overlap_feature], dim = -1), skeleton_edges)
        hskeleton = torch.cat([readout_q, overlap_feature], dim = -1)# hq[n_orig_qnode:]
        weight_input = torch.softmax(self.weighter(hskeleton), dim = -2)
        readout_q = torch.sum(readout_q * weight_input, dim = -2)
        if torch.norm(readout_q) == 0: 
            pjlength = readout_q.dot(readout_g)
        else:
            pjlength = readout_q.dot(readout_g) / torch.norm(readout_q)
        readout = (self.aggregate2(torch.cat([readout_q, readout_g], dim = -1)))
        out1 = self.estimate(readout) *  nonlinear_func(self.weighter2(torch.cat([readout, (xg.shape[0] * torch.ones([1]).to(xq.device))], dim = -1)))
        if self.args.no_interaction: return out1, (0, 0, 0), pjlength
        return out1, (x1s,x2s, ys), pjlength
    
