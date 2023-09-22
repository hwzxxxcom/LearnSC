import torch
from aggregators import *
from decomposers import *
from gnns import GIN
from utils import *
from interactors import *

class Swish(torch.nn.Module):
    def __init__(self,inplace=False):
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
    def __init__(self, args, input_size, model_size, nlayer = 1, max_card = 20, min_card = 0):
        super().__init__()
        self.args = args
        self.test = False
        self.max_card = max_card
        self.min_card = min_card
        self.model_size = model_size
        self.graph_gnn = GIN(input_size, model_size, nlayer, args.device)
        self.query_gnn1 = GIN(input_size, model_size, 1, args.device)
        self.query_gnn2 = GIN(model_size, model_size, nlayer - 1, args.device)
        #self.linear = torch.nn.Linear(model_size, model_size)
        self.linear1 = torch.nn.Linear(model_size, model_size)
        self.linear2 = torch.nn.Linear(model_size, model_size)
        self.linear3 = torch.nn.Linear(model_size, model_size)
        #self.linear4 = torch.nn.Linear(model_size, model_size)
        self.interactor = GIN(model_size, model_size, 1, args.device)
        # self.interactor1 = Interactor(model_size, model_size, args.device)
        self.aggregate = GIN(2 * model_size, model_size, 1, args.device)
        #
        #self.norm = torch.nn.Sequential(torch.nn.Linear(model_size, model_size), swish, torch.nn.Linear(model_size, model_size), swish)
        #self.trans= torch.nn.Sequential(torch.nn.Linear(model_size, model_size), swish, torch.nn.Linear(model_size, model_size), swish)
        #
        self.aggregate2 = torch.nn.Linear(2 * model_size, model_size)
        #self.estimate = torch.nn.Linear(model_size, 1)
        self.weighter = torch.nn.Linear(2 * model_size, 1)
        self.weighter2 = torch.nn.Linear(model_size + 1, 1)
    def forward(self, xg: torch.Tensor, eg: torch.Tensor, n_orig_gnode: int, match:dict[int,int], 
                      xq: torch.Tensor, eq: torch.Tensor, n_orig_qnode: int, overlap: dict[(int,int),list[int]], subqueries: list[list[int]], itedge, npairs):
        hg, hq = self.graph_gnn(xg, eg), self.query_gnn1(xq, eq)
        if self.args.test_plane:
            norms = self.norm(hq)
            hq = self.trans(hq)
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
        lst = list(range(len(itedge[1])))
        random.shuffle(lst)
        itedge = (itedge[:, lst[:5 * n_orig_qnode]])
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
            if self.args.test_plane:
                embedpicker2 = torch.eye(norms.shape[0])
                lst = list(range(n_orig_qnode))
                lst_s = nshuffle(lst)
                npair2 = torch.tensor([lst[:len(lst_s)], lst_s])
                neg_pair2 = embedpicker2[npair2].detach()
                #print(embedpicker.shape)
                N3 = neg_pair2[0].detach()
                N4 = neg_pair2[1].detach()
                #print(N1.shape, N3.shape)
                x1s2 = N3.mm(norms)
                x2s2 = N4.mm(norms)
                x1s = torch.cat([x1s, x1s2], dim = -2)
                x2s = torch.cat([x2s, x2s2], dim = -2)
                ys = torch.cat([ys, -torch.ones(len(neg_pair2[0]))])
                dist_1 = distance_to_plain(norms, embedpicker[:norms.shape[0]].detach().mm(itgraph))
                dist_2 = distance_to_plain(embedpicker2[itedge[0]].detach().mm(norms), embedpicker[itedge[1]].detach().mm(itgraph))
                plane = torch.cat([dist_1, dist_2], dim = 0)
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
        readout = torch.stack([torch.mean(self.linear1(hq[subquery]), dim = -2) for subquery in subqueries], dim = 0)
        readout_q = readout
        readout_g = torch.mean(self.linear2(hg[:n_orig_gnode]), dim = -2)
        overlap_feature = torch.zeros(readout.shape, dtype = torch.float).to(xq.device)
        overlap_count = torch.ones((readout.shape[0], 1)).to(xq.device)
        skeleton_edges = []
        for i, j in overlap:
            if i > j: continue
            skeleton_edges.append((i,j))
            overlap_feature[i] += torch.mean(self.linear3(hq[overlap[(i, j)]]), dim = 0)
            overlap_count[i][0] += 1
            overlap_feature[j] += torch.mean(self.linear3(hq[overlap[(i, j)]]), dim = 0)
            overlap_count[j][0] += 1
        overlap_count = overlap_count.detach()
        overlap_feature = overlap_feature / overlap_count
        skeleton_edges = (torch.tensor(skeleton_edges).reshape((2, -1)).type(torch.long)).to(xg.device)# + torch.tensor(n_orig_qnode)).to(xg.device)
        overlap_feature = self.aggregate(torch.cat([readout_q, overlap_feature], dim = -1), skeleton_edges)
        hskeleton = torch.cat([readout_q, overlap_feature], dim = -1)# hq[n_orig_qnode:]
        weight_input = torch.softmax(self.weighter(hskeleton), dim = -2)
        hskeleton = readout_q * weight_input
        readout_q = torch.mean(hskeleton, dim = -2)
        # if torch.norm(readout_q) == 0: 
        #     pjlength = readout_q.dot(readout_g)
        # else:
        # 
        nskeleton = torch.norm(hskeleton, dim = -1, keepdim = True)
        if torch.any(nskeleton == 0): 
            pjlength = torch.zeros_like(hskeleton.dot(readout_g)) 
        else:
            pjlength = torch.sum(hskeleton * readout_g.reshape((1, -1)), dim = -1) / torch.norm(hskeleton, dim = -1).to(xg.device)
        readout = (self.aggregate2(swish(torch.cat([readout_q, readout_g], dim = -1))))
        out = self.weighter2(torch.cat([readout, (xg.shape[0] * torch.ones([1]).to(xq.device))], dim = -1))
        out1 = nonlinear_func(out, self.max_card, self.min_card)
        if self.args.test_plane: return out1, (x1s,x2s, ys), plane
        if self.args.no_interaction: return out1, (0, 0, 0), pjlength
        return out1, (x1s,x2s, ys), pjlength
    
