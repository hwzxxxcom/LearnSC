import torch
import time
import networkx as nx
import numpy as np
import math
import igraph as ig
import argparse
import random
import os
from tqdm import tqdm
import faulthandler
from aggregators import *
from decomposers import *
from gnns import GIN
from utils import *
from interactors import *

faulthandler.enable()
device = 'cuda:0'
logfile = ''
parser = argparse.ArgumentParser()

parser.add_argument('--in-feat', type=int, default=64,
                    help='input feature dim')
parser.add_argument('--model-feat', type=int, default=128)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--no-direction-embed', action='store_true')
parser.add_argument('--dirtrain-interval', type = int, default=2, help='how many epoch between two turns of direction learning')
parser.add_argument('--no-length-embed', action='store_true')
parser.add_argument('--no-query-decomposition', action='store_true')
parser.add_argument('--lentrain-interval', type = int, default=2, help='how many epoch between two turns of pjlength learning')
parser.add_argument('--negtive-samples', type=int, default=4)
parser.add_argument('--neg-cut', type=int, default=100)
parser.add_argument('--no-ndec', action='store_true', help='用NeurSC的substructure分解方法')
parser.add_argument('--no-interaction', action='store_true', help='用NeurSC的substructure分解方法')
parser.add_argument('--dataname', type=str, default='iwiki')
parser.add_argument('--input-size', type=int, default=14)
parser.add_argument('--n-query-node', type = str, default='all')
parser.add_argument('--training-rate', type = float, default=0.8)


args = parser.parse_args()
print(args)

def gettime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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

class Nagy(torch.nn.Module):
    def __init__(self, input_size, model_size, nlayer = 2):
        super().__init__()
        self.test = False
        self.model_size = model_size
        self.graph_gnn = GIN(input_size, model_size, nlayer)
        self.query_gnn1 = GIN(input_size, model_size, 1)
        self.query_gnn2 = GIN(model_size, model_size, nlayer - 1)
        self.linear = torch.nn.Linear(model_size, model_size)
        self.linear1 = torch.nn.Linear(model_size, model_size)
        self.linear2 = torch.nn.Linear(model_size, model_size)
        self.linear3 = torch.nn.Linear(model_size, model_size)
        self.linear4 = torch.nn.Linear(model_size, model_size)
        self.interactor = GIN(model_size, model_size, 2)
        self.interactor1 = Interactor(model_size, model_size)
        self.aggregate = GIN(2 * model_size, model_size, 2)
        self.aggregate2 = torch.nn.Linear(2 * model_size, model_size)
        self.estimate = torch.nn.Linear(model_size, 1)
        self.weighter = torch.nn.Linear(2 * model_size , 1)
        self.weighter2 = torch.nn.Linear(model_size + 1, 1)
    def forward(self, xg: torch.Tensor, eg: torch.Tensor, n_orig_gnode: int, match:dict[int,int], 
                      xq: torch.Tensor, eq: torch.Tensor, n_orig_qnode: int, overlap: dict[(int,int),list[int]], subqueries: list[list[int]], itedge, npairs):
        hg, hq = self.graph_gnn(xg, eg), self.query_gnn1(xq, eq)
        transg = torch.zeros((n_orig_gnode, hg.shape[0]))
        transg[:, :n_orig_gnode] = torch.eye(n_orig_gnode)
        transg = transg.detach()
        orig_hg = transg.mm(hg)
        transq = torch.zeros((n_orig_qnode, hq.shape[0]))
        transq[:, :n_orig_qnode] = torch.eye(n_orig_qnode)
        transq = transq.detach()
        orig_hq = transq.mm(hq)
        transq1 = torch.zeros((n_orig_qnode + n_orig_gnode, n_orig_qnode))
        transq1[:n_orig_qnode, :n_orig_qnode] = torch.eye(n_orig_qnode)
        transg1 = torch.zeros((n_orig_qnode + n_orig_gnode, n_orig_gnode))
        transg1[n_orig_qnode:, :] = torch.eye(n_orig_gnode)
        itgraph = transq1.mm(orig_hq) + transg1.mm(orig_hg)
        if args.no_interaction:
            itgraph = self.interactor(itgraph, torch.LongTensor().reshape((2,0)))
        else:
            itgraph = self.interactor(itgraph, itedge)
            embedpicker = torch.eye(itgraph.shape[0])
            npair = torch.tensor(npairs, dtype = torch.long).T if npairs else torch.LongTensor().reshape(2,0)
            pos_pair = embedpicker[itedge]
            neg_pair = embedpicker[npair]
            P1 = pos_pair[0].detach()
            P2 = pos_pair[1].detach()
            N1 = neg_pair[0].detach()
            N2 = neg_pair[1].detach()
            X1 = torch.cat((P1, N1), dim = 0).detach()
            X2 = torch.cat((P2, N2), dim = 0).detach()
            x1s = X1.mm(itgraph)
            x2s = X2.mm(itgraph)
            ys = torch.cat((torch.ones(pos_pair.shape[1]), -torch.ones(neg_pair.shape[1])), dim = -1)
        gmsk = torch.tensor([[1]]*n_orig_gnode+[[0]]*(hg.shape[0]-n_orig_gnode),dtype=torch.float)
        qmsk = torch.tensor([[1]]*n_orig_qnode+[[0]]*(hq.shape[0]-n_orig_qnode),dtype=torch.float)
        hg = (1-gmsk) * hg + transg.T.mm(transg1.T.mm(itgraph))
        hq = (1-qmsk) * hq + transq.T.mm(transq1.T.mm(itgraph))
        match_count = dict()
        mask, values = torch.ones([hq.shape[0], 1], dtype=torch.float).to(args.device), torch.zeros(hq.shape, dtype=torch.float).to(args.device).requires_grad_(False)
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
        if args.no_interaction: return out1, (0,0, 0), pjlength
        return out1, (x1s,x2s, ys), pjlength
    


def train(model, data, interedges, cdicts, cardinalities, testnames, testdata, testinteredge, testcardinalities, lr = 0.00005):
    global pred, card 
    evallosses = []
    model.train()
    npairs = []
    for i in tqdm(range(len(data))):
        npairs.append([])
        for j in range(len(data[i][1])):
            all_dnodes = set(range(data[i][0][2], data[i][1][j][2]))
            npairs[i].append([])
            for qv in range(len(cdicts[i][j])):
                npairs[i][j].extend((qv, dv) for dv in (all_dnodes - cdicts[i][j][qv]))
            random.shuffle(npairs[i][j])
            npairs[i][j] = npairs[i][j][:args.neg_cut]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_func = torch.nn.MSELoss()
    dloss_func = torch.nn.CosineEmbeddingLoss(0.3, reduction = 'mean')
    lloss_func = torch.nn.L1Loss()
    for epoch in range(100):
        rdm = list(range(len(data)))
        if epoch % 5 != 0: random.shuffle(rdm)
        data = [data[i] for i in rdm]
        interedges = [interedges[i] for i in rdm]
        npairs = [npairs[i] for i in rdm]
        cardinalities = cardinalities[rdm]
        content = '[%s] epoch: %d/100' % (gettime(), epoch)
        print(content)
        print(content, file = logfile)
        total = 0.
        count = 0
        for batch in range(math.ceil(len(data)/args.batch_size)):
            predlst = []
            dircount = 0 
            dloss = 0
            lenlst = []
            card = cardinalities[batch * args.batch_size: (batch + 1) * args.batch_size]
            for ((xq, eq, n_orig_qnode, overlap, subqueries, query_name), substructures), itedges, nprs in zip(data[batch * args.batch_size: (batch + 1) * args.batch_size], interedges[batch * args.batch_size: (batch + 1) * args.batch_size], npairs[batch * args.batch_size: (batch + 1) * args.batch_size]):
                predi = torch.tensor([0], dtype = torch.float).to(args.device)
                leni = torch.tensor([0], dtype = torch.float).to(args.device)
                lencount = 0 
                #print('  %d'%len(substructures))
                for (xg, eg, n_orig_gnode, match), itedge, npr in zip(substructures, itedges, nprs):
                    #print(len(match))
                    estimation4substructure, emb_info, pjlength = model(xg, eg, n_orig_gnode, match, xq, eq, n_orig_qnode, overlap, subqueries, itedge, npr)
                    predi += estimation4substructure
                    if not args.no_direction_embed and epoch % args.dirtrain_interval == 0:
                        x1,x2,y = emb_info
                        if y.shape[0] != 0:
                            dloss += dloss_func(x1,x2,y) 
                            dircount += 1
                    if not args.no_length_embed and epoch % args.lentrain_interval == 0:
                        leni += pjlength
                        lencount += 1
                predlst.append(predi)
                lenlst.append(leni/lencount)
            if not predlst: continue
            pred = torch.log(torch.relu(torch.stack(predlst)) + 1).to(args.device)
            lens = torch.stack(lenlst).to(args.device)
            if epoch >= 5:
                if ((pred==0) * torch.ones_like(pred)).reshape(-1).sum() >= pred.shape[0] * 0.7: return None
            assert card.shape == pred.shape
            optimizer.zero_grad()
            reg_loss = loss_func(pred, card) * len(pred) / args.batch_size
            if not args.no_length_embed and epoch % args.lentrain_interval == 0:
                lloss = lloss_func(lens, card)
            else: lloss = 0
            if epoch < 10:
                loss = dloss * 0. + lloss * 0. + reg_loss * 1
            elif epoch < 40:
                loss = dloss * 0.2 + lloss * 0.2 + reg_loss * 0.6 
            else:
                loss = dloss * 0.2 + lloss * 0.2 + reg_loss * 0.6 
            loss.backward()
            optimizer.step()
            total += float(loss.detach() * len(pred))
            count += len(pred)
            if batch % 10 == 0:
                print(pred.T, file=logfile)
                print(card.T, file=logfile)
                print('[%s] - e: %d, b: %d, ls: %.4f, dls = %.4f, rls = %.4f, lls = %.4f' % (gettime(), epoch, batch, loss.cpu().detach(), dloss, reg_loss, lloss))
                print('[%s] - e: %d, b: %d, ls: %.4f, dls = %.4f, rls = %.4f, dct = %d' % (gettime(), epoch, batch, loss.cpu().detach(), dloss, reg_loss, dircount), file = logfile)
        if count != 0: 
            print('- epoch: %d, mean_loss: %.4f' % (epoch, total/ count))
            print('[%s] - epoch: %d, mean_loss: %.4f' % (gettime(), epoch, total/ count), file = logfile)
        if epoch % 1 == 0:
            print('Evaluating')
            print('[%s] Evaluating' % gettime(), file = logfile)
            evallosses.append(evaluate(model, testdata, testinteredge, testcardinalities, prt = True if epoch % 5 == 0 else False))
    return model

def evaluate(model, data, interedges, cardinalities, prt):
    global pred, card
    model.eval()
    #args.batch_size=16
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    #leakyrelu = torch.nn.LeakyReLU(0.2)
    preds = torch.zeros(0).reshape((0,1))
    card = cardinalities
    for i in range(math.ceil(len(data) / args.batch_size)):
        predlst = []
        for ((xq, eq, n_orig_qnode, overlap, subqueries, query_name), substructures), itedges in zip(data[i * args.batch_size: (i + 1) * args.batch_size], interedges[i * args.batch_size: (i + 1) * args.batch_size]):
            predi = torch.tensor([0], dtype = torch.float).to(args.device)
            #print('  %d'%len(substructures))
            for (xg, eg, n_orig_gnode, match), itedge in zip(substructures, itedges):
                estimation4substructure, _, _ = model(xg, eg, n_orig_gnode, match, xq, eq, n_orig_qnode, overlap, subqueries, itedge, [])
                predi += estimation4substructure.detach()
                del estimation4substructure
            predlst.append(predi)
        pred = torch.log(torch.relu(torch.stack(predlst)) + 1).to(args.device).detach()
        del predlst
        preds = torch.cat([preds, pred], dim = 0)
    pred = preds
    assert card.shape == pred.shape
    #print(pred)
    #card = card * pred.detach().bool()
    loss = loss_func(pred, card)
    if prt: print(pred.T)
    if prt: print(card.T)
    if prt: print(pred.T, file = logfile)
    if prt: print(card.T, file = logfile)
    if prt: print('- evaluate, %s with N = %s, loss: %.4f' % (args.dataname, args.n_query_node, loss.cpu().detach()))
    if prt: print('[%s] - evaluate, loss: %.4f' % (gettime(), loss.cpu().detach()), file = logfile)
    return float(loss.cpu().detach())

if __name__ == "__main__":
    graphname = args.dataname#'iwiki'#'wordnet18rr' #'citeseer'#'iyeast' #'facebook' #'yeast' # 
    query_node_number = args.n_query_node
    logfile = open('../log/NEWNAGY_%s_%s.txt' % (graphname, query_node_number), 'a')
    queries, query_files = [], []
    #query_file_names = set(os.listdir('/home/nagy/data/NeurSC/citeseer/queries_16/'))
    cardinalities = dict()
    if query_node_number == 'all':
        nnodelst = [5,10,20,40,80]
        nnodelst = [i for i in nnodelst if 'cardinalities_%d.lst' % i in os.listdir('/home/nagy/data/NeurSC/%s/' % graphname)]
    else:
        nnodelst = list(map(int, query_node_number.strip().split(',')))
    for nnode in nnodelst:
        with open('/home/nagy/data/NeurSC/%s/cardinalities_%d.lst' % (graphname, nnode)) as card_file:
            for ind, line in enumerate(card_file):
                line = line.strip()
                i = int(line[6:11])
                if not line: break
                name, card = line.split()
                name = ('queries_%d/' % nnode) + name
                cardinalities[name] = int(card)
        #print(cardinalities.keys())
    for nnode in nnodelst:
        for i in range(300):
            if not 'queries_%d/query_%05d.grf' % (nnode, i) in cardinalities: continue            
            assert 'queries_%d/query_%05d.grf' % (nnode, i) in cardinalities
            # query_file_names = set(os.listdir('/home/nagy/data/NeurSC/citeseer/queries_16/'))
            # if not 'query_%05d.grf' % i in query_file_names: continue 
            queries.append(load_grf(('/home/nagy/data/NeurSC/%s/queries_%d/query_%05d.grf' % (graphname, nnode, i)) , False)) #, load_grf('../data/toy/queries/query_00001.grf', False)]
            query_files.append('/home/nagy/data/NeurSC/%s/queries_%d/query_%05d.grf' % (graphname, nnode, i))
    cardinalities = torch.log(torch.tensor([cardinalities[n] for n in ['queries_%d/query_%05d.grf' % (nnode, i) for nnode in nnodelst  for i in range(len(queries))]], dtype = torch.float)).reshape([-1, 1]).to(args.device)
    grfinfo = [(nnode, i) for nnode in nnodelst  for i in range(300)]
    graph = load_grf('/home/nagy/data/NeurSC/%s/data.grf' % graphname, False)
    if args.no_query_decomposition:
        qdecomposer = NoQDecomposer()
    else:
        qdecomposer = ToyQDecomposer()
    #
    if args.no_ndec:
        gdecomposer = GQLGDecomposer(graphname)
    else:
        gdecomposer = GQLGDecomposerN(graphname)
    data = []
    # wordnet 16 wiki_c 14 yeast 14 citeseer 6
    embedder = torch.eye(args.input_size, dtype=torch.float)
    querynames = []
    interedges = []
    candidicts = []
    for query, query_name in tqdm(list(zip(queries, query_files))):
        #print('p ' + query_name)
        #print('[%s] p ' % gettime() + query_name, file = logfile)
        dt, iteg, cdct = preprocess(args, gdecomposer, qdecomposer, graph, query, embedder, query_name)
        # ((query_feature.to(args.device), query_edge.to(args.device), n_orig_qnode, overlap, subqueries, query_name), substructures)
        # substructures.append((substructure_feature.to(args.device), substructure_edge.to(args.device), n_orig_gnode, match2query))
        data.append(dt)
        interedges.append(iteg)
        candidicts.append(cdct)
        tag = True
        #if not data[-1][-1]:
        #    print('↑↑ ', end = '')
        #    print('↑↑ ', end = '', file = logfile)
        for i in range(len(data[-1][-1])):
            if data[-1][-1][i][-1]: tag = False
        #if tag: 
        #    print('↑ ', end = '')
        #    print('↑ ', end = '', file = logfile)
        querynames.append(query_name.strip().split('/')[-1])
    lst = list(range(len(data)))
    random.seed(0)
    random.shuffle(lst)
    bder = math.floor(len(data) * args.training_rate)
    traingrfinfo        = [grfinfo[i]       for i in lst[:bder]] 
    testgrfinfo         = [grfinfo[i]       for i in lst[bder:]] 
    traininteredge      = [interedges[i]    for i in lst[:bder]] 
    testinteredge       = [interedges[i]    for i in lst[bder:]] 
    traincdict          = [candidicts[i]    for i in lst[:bder]] 
    testcdict           = [candidicts[i]    for i in lst[bder:]] 
    traindata           = [data[i]          for i in lst[:bder]]
    testdata            = [data[i]          for i in lst[bder:]]
    traincardinalities  = cardinalities[lst[:bder]]
    testcardinalities   = cardinalities[lst[bder:]]
    trainnames          = [querynames[i]    for i in lst[:bder]]
    testnames           = [querynames[i]    for i in lst[bder:]]
    
    # yeast 14 citeseer 6
    while True:
        model = Nagy(args.input_size, 128).to(args.device)
        if train(model, traindata, traininteredge, traincdict, traincardinalities, testgrfinfo, testdata, testinteredge, testcardinalities, 0.00005): break
    #logfile = open('../log/noelec_ablation_neurscinteract_%s_%d.txt' % (graphname, query_node_number), 'a')
    #torch.save(model.state_dict(), 'model_state_dict.pth')
    #model.load_state_dict(torch.load("model_state_dict.pth"))
    #model.load_state_dict(torch.load('nagywkseed2.pth'))
    evaluate(model, testdata, testinteredge, testcardinalities, True)
    logfile.close()
    torch.save(model.state_dict(), '../nagy_models/modelsmodel_state_dict_%s_%s.pth'%(graphname, query_node_number))
    #raise
