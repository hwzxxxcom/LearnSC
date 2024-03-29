import argparse
import faulthandler
import math
import os
import time
import torch
import random

from model import LearnSC
from decomposers import *
from tqdm import tqdm
from utils import *

faulthandler.enable()

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
parser.add_argument('--no-ndec', action='store_true')
parser.add_argument('--no-interaction', action='store_true')
parser.add_argument('--dataname', type=str, default='iiyeast')
parser.add_argument('--input-size', type=int, default=71)
parser.add_argument('--n-query-node', type = str, default='5')
parser.add_argument('--training-rate', type = float, default=0.9)
parser.add_argument('--test-plane', action='store_true')

args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else args.device

print(args)

def gettime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def train(model, data, interedges, cdicts, cardinalities, testnames, testdata, testinteredge, testcardinalities, lr = 0.00002):
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
    lloss_func = torch.nn.L1Loss(reduction='mean')
    ploss_func = torch.nn.L1Loss(reduction='sum')
    for epoch in range(200):
        rdm = list(range(len(data)))
        if epoch % 5 != 0: random.shuffle(rdm)
        data = [data[i] for i in rdm]
        interedges = [interedges[i] for i in rdm]
        cdicts = [cdicts[i] for i in rdm]
        npairs = [npairs[i] for i in rdm]
        cardinalities = cardinalities[rdm]
        content = '[%s] epoch: %d/200' % (gettime(), epoch)
        print(content)
        print(content, file = logfile)
        total = 0.
        count = 0
        for batch in range(math.ceil(len(data)/args.batch_size)):
            lloss = torch.tensor([0.]).to(args.device)
            predlst = []
            dircount = 0 
            dloss = torch.tensor(0, dtype=torch.float).to(args.device)
            ploss = torch.tensor(0, dtype=torch.float).to(args.device)
            pcount = torch.tensor(0, dtype=torch.float).to(args.device)
            lenlst = []
            card = cardinalities[batch * args.batch_size: (batch + 1) * args.batch_size]
            cand = cdicts[batch * args.batch_size: (batch + 1) * args.batch_size]
            for ((xq, eq, n_orig_qnode, overlap, subqueries, query_name), substructures), itedges, nprs, cdcts in zip(data[batch * args.batch_size: (batch + 1) * args.batch_size], interedges[batch * args.batch_size: (batch + 1) * args.batch_size], npairs[batch * args.batch_size: (batch + 1) * args.batch_size], cand):
                predi = torch.tensor([0], dtype = torch.float).to(args.device)
                for (xg, eg, n_orig_gnode, match), itedge, npr, cdct in zip(substructures, itedges, nprs, cdcts):
                    if args.test_plane:
                        estimation4substructure, emb_info, plane = model(xg, eg, n_orig_gnode, match, xq, eq, n_orig_qnode, overlap, subqueries, itedge, npr)
                        predi += estimation4substructure
                        ploss += ploss_func(plane, torch.zeros_like(plane, dtype = torch.float))
                        pcount += plane.shape[0]
                    else:
                        estimation4substructure, emb_info, pjlength = model(xg, eg, n_orig_gnode, match, xq, eq, n_orig_qnode, overlap, subqueries, itedge, npr)
                        predi += estimation4substructure
                        if not args.no_direction_embed and epoch % args.dirtrain_interval == 0:
                            x1,x2,y = emb_info
                            if y.shape[0] != 0:
                                dloss += dloss_func(x1,x2,y) 
                                dircount += 1
                        if not args.no_length_embed and epoch % args.lentrain_interval == 0:
                            true_proj = torch.stack([torch.sum(torch.tensor([torch.log(torch.tensor(len(cdct[node]) + 1)) / len(subquery)  for node in subquery])) for subquery in subqueries]).to(args.device)
                            if random.random() > .9:
                                lloss += lloss_func(pjlength.reshape((-1, 1)), true_proj.reshape((-1, 1)))
                predlst.append(predi)
            if not predlst: continue
            pred = torch.log(torch.relu(torch.stack(predlst)) + 1).to(args.device)
            #pred = torch.relu(torch.stack(predlst)).to(args.device)
            if epoch >= 5:
                if ((pred==0) * torch.ones_like(pred)).reshape(-1).sum() >= pred.shape[0] * 0.7: return None
            assert card.shape == pred.shape
            optimizer.zero_grad()
            reg_loss = loss_func(pred, card) * len(pred) / args.batch_size
            if args.test_plane:
                ploss = ploss / pcount if pcount != 0 else 0
                dloss = 0
            else:
                ploss = 0
            if not args.test_plane and not args.no_length_embed and epoch % args.lentrain_interval == 0:
                lloss = lloss
            else: lloss = 0
            if epoch < 10:
                loss = dloss * 0. + lloss * 0. + reg_loss * 1
            elif epoch < 40:
                loss = dloss * 0.2 + lloss * 0.2 + ploss * 0.2 + reg_loss * 0.6 
            else:
                loss = dloss * 0.1 + lloss * 0.1 + ploss * 0.1 + reg_loss * 0.8 
            loss.backward()
            optimizer.step()
            total += float(loss.detach() * len(pred))
            count += len(pred)
            if batch % 10 == 0:
                print(pred.T, file=logfile)
                print(card.T, file=logfile)
                print('[%s] - e: %d, b: %d, ls: %.4f, pls = %.4f, dls = %.4f, rls = %.4f, lls = %.4f' % (gettime(), epoch, batch, loss.cpu().detach(), ploss, dloss, reg_loss, lloss))
                print('[%s] - e: %d, b: %d, ls: %.4f, pls = %.4f, dls = %.4f, rls = %.4f, dct = %d' % (gettime(), epoch, batch, loss.cpu().detach(), ploss, dloss, reg_loss, dircount), file = logfile)
        if count != 0: 
            print('- epoch: %d, mean_loss: %.4f' % (epoch, total/ count))
            print('[%s] - epoch: %d, mean_loss: %.4f' % (gettime(), epoch, total/ count), file = logfile)
        if epoch % 1 == 0:
            print('[%s] Evaluating' % gettime(), file = logfile)
            evallosses.append(evaluate(model, testdata, testinteredge, testcardinalities, prt = True if epoch % 5 == 0 else False))
    return model

def evaluate(model, data, interedges, cardinalities, prt):
    global pred, card
    model.eval()
    loss_func = torch.nn.MSELoss()
    preds = torch.zeros(0).reshape((0,1)).to(args.device)
    card = cardinalities
    for i in range(math.ceil(len(data) / args.batch_size)):
        predlst = []
        for ((xq, eq, n_orig_qnode, overlap, subqueries, query_name), substructures), itedges in zip(data[i * args.batch_size: (i + 1) * args.batch_size], interedges[i * args.batch_size: (i + 1) * args.batch_size]):
            predi = torch.tensor([0], dtype = torch.float).to(args.device)
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
    loss = loss_func(pred, card)
    if prt: print(pred.T)
    if prt: print(card.T)
    if prt: print(pred.T, file = logfile)
    if prt: print(card.T, file = logfile)
    if prt: print('- evaluate, %s with N = %s, loss: %.4f' % (args.dataname, args.n_query_node, loss.cpu().detach()))
    if prt: print('[%s] - evaluate, loss: %.4f' % (gettime(), loss.cpu().detach()), file = logfile)
    return float(loss.cpu().detach())

if __name__ == "__main__":
    graphname = args.dataname
    query_node_number = args.n_query_node
    logfile = open('../log/LearnSC_%s_%s.txt' % (graphname, query_node_number), 'a')
    queries, query_files = [], []
    cardinalities = dict()
    if query_node_number == 'all':
        nnodelst = [5,10,20,40,60]
        nnodelst = [i for i in nnodelst if 'cardinalities_%d.lst' % i in os.listdir('../data/%s/' % graphname)]
    else:
        nnodelst = list(map(int, query_node_number.strip().split(',')))
    for nnode in nnodelst:
        with open('../data/%s/cardinalities_%d.lst' % (graphname, nnode)) as card_file:
            for ind, line in enumerate(card_file):
                line = line.strip()
                i = int(line[6:11])
                if not line: break
                name, card = line.split()
                name = ('queries_%d/' % nnode) + name
                cardinalities[name] = int(card)
    for nnode in nnodelst:
        for i in range(3000):
            if not 'queries_%d/query_%05d.grf' % (nnode, i) in cardinalities: continue         
            queries.append(load_grf(('../data/%s/queries_%d/query_%05d.grf' % (graphname, nnode, i)) , False)) 
            query_files.append('../data/%s/queries_%d/query_%05d.grf' % (graphname, nnode, i))
    carddict = cardinalities
    cardinalities = torch.log(torch.tensor([cardinalities[n] for n in ['queries_%d/query_%05d.grf' % (nnode, i) for nnode in nnodelst  for i in range(3000)] if n in cardinalities], dtype = torch.float)).reshape([-1, 1]).to(args.device)
    grfinfo = [(nnode, i) for nnode in nnodelst  for i in range(3000) if 'queries_%d/query_%05d.grf' % (nnode, i) in carddict]
    graph = load_grf('../data/%s/data.grf' % graphname, False)
    if args.no_query_decomposition:
        qdecomposer = NoQDecomposer()
    else:
        qdecomposer = RdmQDecomposer()
    if args.no_ndec:
        gdecomposer = GQLGDecomposer(graphname)
    else:
        gdecomposer = GQLGDecomposerN(graphname)
    data = []
    datastime = time.time()
    embedder = torch.eye(args.input_size, dtype=torch.float)
    querynames = []
    interedges = []
    candidicts = []
    for query, query_name in tqdm(list(zip(queries, query_files))[:300]):
        #print(query_name)
        dt, iteg, cdct = preprocess(args, gdecomposer, qdecomposer, graph, query, embedder, query_name)
        data.append(dt)
        interedges.append(iteg)
        candidicts.append(cdct)
        tag = True
        for i in range(len(data[-1][-1])):
            if data[-1][-1][i][-1]: tag = False
        querynames.append(query_name.strip().split('/')[-1])
    print(time.time() - datastime)
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

    while True:
        model = LearnSC(args, args.input_size, 128, nlayer=1, max_card = float(traincardinalities.max()) + 1, min_card = max(0, float(traincardinalities.min()) - 1)).to(args.device)
        if train(model, traindata, traininteredge, traincdict, traincardinalities, testgrfinfo, testdata, testinteredge, testcardinalities, 0.0005): break
    evaluate(model, testdata, testinteredge, testcardinalities, True)
    logfile.close()
    torch.save(model.state_dict(), '../saved_models/LearnSC_state_dict_%s_%s.pth'%(graphname, query_node_number))
