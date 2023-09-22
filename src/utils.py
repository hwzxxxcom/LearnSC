import networkx as nx
import igraph as ig
import numpy as np
from filtering import Filtering
import os
import math
import random
import torch
from decomposers import *
from collections import defaultdict
from copy import deepcopy

def nshuffle(lst):
    res = []
    cands = set(lst)
    for i in lst:
        if len(cands) == 1 and i in cands: return res
        res.append(random.choice(list(cands - {i})))
        cands -= {res[-1]}
    return res


def distance_to_plain(norm, point):
    assert norm.dim() == point.dim() == 2
    return torch.sum(norm * point, dim = -1) / torch.sum(norm * norm, dim = -1) ** .5

def log_qerror(pred, card):
    pred = pred.reshape(-1)
    card = card.reshape(-1)
    return np.sign(pred - card) * np.log10(torch.cat([pred/card, card/pred], axis = -1)).max(-1).values

def pred2lgqe(pred, card):
    return np.sign(pred - card) * np.log10(np.stack([(pred + 1) / (card + 1), (card + 1) / (pred + 1)])).max(0)

def qe2lgqe(qe):
    return np.sign(qe) * np.log10(np.abs(qe))

def preprocess_decompose_and_match(gdec: GDecomposer, qdec: QDecomposer, graph: nx.Graph, query: nx.Graph, query_name=None, match_rate=0.1, max_match = 2000):
    QS = QuickSampler()
    subqueries = qdec.decompose(query)
    substructures, candidate_info = gdec.decompose(graph, query, query_name=query_name)
    overlap = dict()
    for i in range(len(subqueries)):
        for j in range(len(subqueries)):
            if i == j: continue
            ovlp = sorted(set(subqueries[i]).intersection(subqueries[j]))
            if ovlp: overlap[(i, j)] = ovlp
    matches = dict() 
    nxsubqueries = []
    nxsubstructures = []
    for squery in subqueries:
        nxsubqueries.append(nx2nx(nx.subgraph(query, squery)))
    for i, (sstructure, mapping) in enumerate(substructures):
        matches[i] = dict()
        for j, squery in enumerate(nxsubqueries):
            break
            mtches = QS.sample(squery, sstructure, r = 0.4, k = 0.02)
            if mtches:
                selected = np.random.choice(list(range(len(mtches))), min(math.ceil(len(mtches) * match_rate), math.ceil(math.log(len(mtches) + 1) * 10) + 1, max_match))
                matches[i][j] = [mtches[i] for i in range(len(mtches))]
    return subqueries, substructures, overlap, matches, candidate_info

def preprocess_add_query_nodes(query: nx.Graph, subqueries: list[list[int]],):
    n_orig_node = len(query.nodes)
    nquery = query.copy()
    for i, subquery in enumerate(subqueries):
        nquery.add_node(len(nquery.nodes), l = -1)
    return nquery, n_orig_node

def preprocess_add_graph_nodes(graph, substructure, matches: dict[int, list[list[int]]]):
    substructure = nx2nx(substructure)
    n_orig_node = len(substructure.nodes)
    nsubstructure = substructure.copy()
    match2query = dict()
    ncount = len(substructure.nodes)
    for matched_subquery in matches:
        mtches = matches[matched_subquery]
        for mtch in mtches:
            nsubstructure.add_node(ncount, l=-1)
            match2query[ncount] = matched_subquery
            for u in mtch:
                nsubstructure.add_edge(u, ncount, l=-1)
            ncount += 1
    return nsubstructure, n_orig_node, match2query

def nx2tensor(graph: nx.Graph, embedder: torch.Tensor, n_orig_node):
    #print('------', n_orig_node, embedder.shape)
    input_size = embedder.shape[-1]
    graph_feature = torch.zeros((len(graph.nodes), input_size), dtype=torch.float)
    #print(len(graph.nodes('l')), list(graph.nodes('l')))
    graph_feature[:n_orig_node] = embedder[torch.tensor(list(graph.nodes('l'))).T[1, :n_orig_node]]
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).T
    if len(graph.edges) == 0:
        return graph_feature, torch.tensor([[],[]], dtype=torch.long)
    return graph_feature, edge_index


def preprocess_query2data(sub_vertices, candidate_info):
    total_len = len(candidate_info)
    num_query_vertex = int(total_len/2)
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(sub_vertices)):
        vertices_dict[sub_vertices[i]] = i + num_query_vertex
    candidate_dict = [set() for i in range(num_query_vertex)]
    for i in range(total_len):
        #print(i)
        if i%2 == 1:
            candidate_list = candidate_info[i].split()
            query_vertex = i//2
            for data_vertex in candidate_list:
                data_vertex = int(data_vertex)
                if data_vertex in sub_vertices:
                    candidate_dict[query_vertex].add(vertices_dict[data_vertex])
                    new_e_u.append(query_vertex)
                    new_e_v.append(vertices_dict[data_vertex])
            try:
                candidate_list = candidate_info[i+2].split()
                data_vertex = int(candidate_list[0])
                if data_vertex in sub_vertices:
                    candidate_dict[query_vertex].add(vertices_dict[data_vertex])
                    new_e_u.append(query_vertex)
                    new_e_v.append(vertices_dict[data_vertex])
            except IndexError:
                continue
    return [new_e_u, new_e_v], candidate_dict

def prepeocess_query2data_batch(sstructures, candidate_info):
    # print(candidate_info)
    total_len = len(candidate_info)
    num_query_vertex = int(total_len/2)
    new_es = [[[], []] for _ in range(len(sstructures))]
    candidate_dicts = [[set() for i in range(num_query_vertex)] for _ in range(len(sstructures))]
    n_origin_data_node = max((max(s[1]) for s in sstructures))
    id_dict = dict() # datav -> ind of sstructure
    for i, (sstructure, mapping) in enumerate(sstructures):
        for d_v in mapping:
            id_dict[d_v] = i
    for i in range(total_len):
        #print(i)
        if i%2 == 1:
            candidate_list = candidate_info[i].split()
            query_vertex = i//2
            for data_vertex in candidate_list:
                data_vertex = int(data_vertex)
                if data_vertex not in id_dict: continue
                ind = id_dict[data_vertex]
                candidate_dicts[ind][query_vertex].add(sstructures[ind][1][data_vertex])
                new_es[ind][0].append(query_vertex)
                new_es[ind][1].append(sstructures[ind][1][data_vertex])
    return new_es, candidate_dicts

import time

def preprocess(args, g_dec: GDecomposer, q_dec: QDecomposer, graph: nx.Graph, query: nx.Graph, embedder: torch.Tensor, query_name = None, matching_rate = 0.1):
    st = time.time()
    subqueries, sstructures, overlap, matches, candidate_info = preprocess_decompose_and_match(g_dec, q_dec, graph, query, query_name, matching_rate)
    nquery, n_orig_qnode = preprocess_add_query_nodes(query, subqueries)
    substructures = []
    #print(time.time() - st)
    interedgess, candidate_dicts = prepeocess_query2data_batch(sstructures, candidate_info)
    interedges = []
    for q2dedges in interedgess:
        itedges = torch.LongTensor(q2dedges)
        assert len(itedges.shape) == 2
        lst = list(range(len(itedges[1])))
        random.shuffle(lst)
        interedges.append(itedges[:, lst[:int(.1*len(lst))]].to(args.device))
    for (sstructure, mapping), ss_index in zip(sstructures, matches):
        nsubstructure, n_orig_gnode, match2query = preprocess_add_graph_nodes(graph, sstructure, matches[ss_index])
        for key in match2query: match2query[key] += n_orig_qnode
        substructure_feature, substructure_edge = nx2tensor(nsubstructure, embedder, n_orig_gnode)
        substructures.append((substructure_feature.to(args.device), substructure_edge.to(args.device), n_orig_gnode, match2query))
        continue
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        alphabet = ''.join(random.sample(alphabet, 16))
        if 'learnsc' not in os.listdir('/dev/shm'): os.system('mkdir /dev/shm/learnsc')
        QuickSampler.save2graph(nss, '/dev/shm/learnsc/g_%s.graph'%alphabet)
        QuickSampler.save2graph(query, '/dev/shm/learnsc/q_%s.graph'%alphabet)
        ginfo = QuickSampler.nx2ginfo(nss, True)
        qinfo = QuickSampler.nx2qinfo(query)
        if 'try':
            tag = True
            if len(ginfo[0]) >= len(qinfo[0]): 
                candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = Filtering(qinfo, ginfo).cpp_GQL('/dev/shm/learnsc/q_%s.graph'%alphabet, '/dev/shm/learnsc/g_%s.graph'%alphabet)
                subgraph_sampler = SampleSubgraph(qinfo, ginfo)
                if args.no_ndec:
                    subgraph_info = subgraph_sampler.load_induced_subgraph2(candidates,induced_subgraph_list, neighbor_offset)
                else:
                    subgraph_info = subgraph_sampler.find_subgraph_reduced(candidate_info, query, nss) 
                if (subgraph_info[0]): 
                    q2dedges, candidate_dict = preprocess_query2data(subgraph_info[0][0], candidate_info)
                    query2data_edge_list = torch.LongTensor(q2dedges)
                    itedges = query2data_edge_list.to(args.device)
                    assert len(itedges.shape) == 2
                    lst = list(range(len(itedges[1])))
                    random.shuffle(lst)
                    interedges.append(itedges[:, lst[:1*len(lst)]])
                    candidate_dicts.append(candidate_dict)
                else: tag = False
            else: tag = False
            if not tag:
                interedges.append(torch.LongTensor([[],[]]).to(args.device))
                candidate_dicts.append([])
        os.system('rm /dev/shm/learnsc/g_%s.graph'%alphabet)
        os.system('rm /dev/shm/learnsc/q_%s.graph'%alphabet)
    
    query_feature, query_edge = nx2tensor(nquery, embedder, n_orig_qnode)
    return ((query_feature.to(args.device), query_edge.to(args.device), n_orig_qnode, overlap, subqueries, query_name), substructures), interedges, candidate_dicts

def load_grf(filepath, directed):
    file = open(filepath)
    nnode = int(file.readline().strip())
    graph = nx.DiGraph() if directed else nx.Graph()
    for v in range(nnode):
        v, l = map(int, file.readline().strip().split())
        graph.add_node(v, l = l)
    for u in range(nnode):
        nedge = int(file.readline().strip())
        for e in range(nedge):
            uu, v, l = map(int, file.readline().strip().split())
            assert u == uu
            graph.add_edge(u, v, l = l)
    return graph

def save_grf(filepath, graph):
    with open(filepath, 'w') as f:
        f.write('%d\n' % len(graph.nodes))
        for v in range(len(graph.nodes)):
            f.write('%d %d\n' % (v, graph.nodes[v]['l']))
        ecount = 0 
        for v in range(len(graph.nodes)):
            stdedges = [vv for vv in sorted(graph.neighbors(v)) if vv > v]
            nv = len ( stdedges)
            f.write('%d\n' % nv)
            for vv in stdedges:
                if v > vv: continue
                ll = graph.edges[v,vv]['l']
                f.write('%d %d %d\n' % (v, vv, ll))

def constructnx(vertices, labels, edges, lmt):
    graph = nx.Graph()
    mapping = dict()
    for i in range(len(vertices)):
        mapping[vertices[i]] = i
        graph.add_node(i, l=labels[vertices[i]])
    for u,v in zip(edges[0], edges[1]):
        if (u, v) not in graph.edges:
            graph.add_edge(mapping[u], mapping[v], l=0)
    return graph, mapping




def nx2nx(graph): # to optimize the subgraph with discontiguous node id
    o2n = dict()
    directed = graph.is_directed()
    if directed:
        ngraph = nx.DiGraph()
        assert ngraph.is_directed()
    else:
        ngraph = nx.Graph()
        assert not ngraph.is_directed()
    for v in graph.nodes:
        o2n[v] = len(ngraph.nodes)
        ngraph.add_node(o2n[v], l=graph.nodes[v]['l'])
    for u, v in graph.edges:
        ngraph.add_edge(o2n[u], o2n[v], l=graph.edges[u, v]['l'])
    return ngraph

def ig2nx(graph):
    directed = graph.is_directed()
    if directed:
        ngraph = nx.DiGraph()
        assert ngraph.is_directed()
    else:
        ngraph = nx.Graph()
        assert not ngraph.is_directed()
    for v in graph.vs: 
        ngraph.add_node(v.index, l=v['l'])
    for e in graph.es:
        u, v = e.tuple
        ngraph.add_edge(u, v, l=e['l'])
    return ngraph

def nx2ig(graph, for_sub = False):
    o2n = dict()
    directed = graph.is_directed()
    if directed:
        iggraph = ig.Graph(directed=True)
        assert iggraph.is_directed()
    else:
        iggraph = ig.Graph(directed=False)
        assert not iggraph.is_directed()
    for v in graph.nodes:
        o2n[v] = len(iggraph.vs) if for_sub else v
        iggraph.add_vertex(o2n[v], l=graph.nodes[v]['l'])
    for u, v in graph.edges:
        iggraph.add_edge(o2n[u], o2n[v], l=graph.edges[u, v]['l'])
    return iggraph

def raise_not_implemented():
    raise NotImplementedError

def nonlinear_func(x, max_card, min_card):
    #return torch.exp(x)    
    return torch.exp(min_card + (max_card - min_card) * (torch.tanh(1/15 * x)/2 + .5) )
    return torch.exp(min_card + (max_card - min_card) * torch.tanh(1/15 * x))
    return torch.exp(30 * torch.tanh(1/15 * x))

    
class QuickSampler():
    def __init__(self):
        self.query = None
        self.graph = None
        self.candidates = None
        self.dfsseq = None
        self.matched = list()
    def search(self, cur = 0):
        u = self.dfsseq[cur]
        for v in self.candidates[u]:
            # TODO: IF u can match to v based on first #cur nodes
            if self.query.nodes[u]['l'] != self.graph.nodes[v]['l']: continue
            tag = True
            for i in range(cur):
                if (u, self.dfsseq[i]) in self.query.edges and (v, self.matched[self.dfsseq[i]]) not in self.graph.edges:
                    tag = False
                    break
            if not tag: continue
            self.matched[u] = v
            if cur == len(self.candidates) - 1: return True
            if self.search(cur + 1): return True
            self.matched[u] = -1
    def dfs(self, query):
        nodes = list(query.nodes)
        dfs = [np.random.choice(nodes)]
        while True:
            if len(dfs) == len(nodes): return dfs
            cand = set()
            for v in dfs: cand = cand.union(query.neighbors(v))
            cand = cand - set(dfs)
            dfs.append(np.random.choice(list(cand)))

    def sample(self, query: nx.Graph, graph: nx.Graph, **para) -> list[list[int]]:
        self.query = query
        self.graph = graph
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        alphabet = ''.join(random.sample(alphabet, 16))
        if 'learnsc' not in os.listdir('/dev/shm'): os.system('mkdir /dev/shm/learnsc')
        g = QuickSampler.save2graph(graph, '/dev/shm/learnsc/g_%s.graph'%alphabet)
        q = QuickSampler.save2graph(query, '/dev/shm/learnsc/q_%s.graph'%alphabet)
        ginfo = QuickSampler.nx2ginfo(graph, True)
        qinfo = QuickSampler.nx2qinfo(query)
        try:
            _, _, _, _, candidate_info = Filtering(qinfo, ginfo).cpp_GQL('/dev/shm/learnsc/q_%s.graph'%alphabet, '/dev/shm/learnsc/g_%s.graph'%alphabet)
        except:
            return [] # candidate_info = ['0', ''] * len(query.nodes)
        candidates = []
        for i in range(len(candidate_info) // 2):
            assert int(candidate_info[i * 2]) == len(candidate_info[i * 2 + 1].strip().split())
            candidates.append(list(map(int, candidate_info[i * 2 + 1].strip().split())))
        #raise
        os.system('rm /dev/shm/learnsc/g_%s.graph'%alphabet)
        os.system('rm /dev/shm/learnsc/q_%s.graph'%alphabet)
        if 'k' in para: k = para['k']
        else: k = 1.0
        if 'r' in para: r = para['r']
        else: r = 0.5
        try:
            N = min(int(np.ceil(k * np.sqrt(np.prod(np.sqrt([len(i) for i in candidates])))) + 5), 2000)
        except:
            N = 5
        '''
        '''
        N = 0
        '''
        '''
        #r = r * ((1 / (1 + 0.2 * np.exp(-N + 10)))) 
        F = int(len(query.nodes) * r)
        #print('N = %d, F = %d' % (N, F), end = ', ')
        res = []
        for _ in range(N):
            # construct new candidat set
            self.matched = [-1 for _ in query.nodes]
            self.dfsseq = self.dfs(query)
            self.candidates = deepcopy(candidates)
            for i in range(len(query.nodes)):
                v = self.dfsseq[i]
                if len(self.candidates[v]) == 0:
                    #print('warning')
                    continue
                if i < F:
                    self.candidates[v] = [np.random.choice(self.candidates[v])]
                else:
                    random.shuffle(self.candidates[v])
            #search
            self.search()
            if -1 not in self.matched: res.append(self.matched[:]) 
        return res

    @staticmethod
    def save2graph(graph: nx.Graph, path):
        if 'g_' in path:
            ecount = [0 for _ in graph.nodes]
            tte = 0
            for u, v in graph.edges:
                if u == v: continue
                ecount[u] += 2
                ecount[v] += 2
                tte += 2
            with open(path, 'w') as ofile:
                ofile.write('t %d %d\n' % (len(graph.nodes), tte))
                for u in graph.nodes:
                    ofile.write('v %d %d %d\n' % (u, graph.nodes[u]['l'], ecount[u]))
                for u, v in graph.edges:
                    if u == v: continue
                    ofile.write('e %d %d\n' % (u, v))
                    ofile.write('e %d %d\n' % (v, u))
        else:
            with open(path, 'w') as ofile:
                ofile.write('t %d %d\n' % (len(graph.nodes), len(graph.edges)))
                for u in graph.nodes:
                    ofile.write('v %d %d %d\n' % (u, graph.nodes[u]['l'], graph.degree[u]))
                for u, v in graph.edges:
                    if u == v: continue
                    ofile.write('e %d %d\n' % (u, v))


    @staticmethod
    def nx2ginfo(graph: nx.Graph, tag = False):
        g_nid = list(graph.nodes)
        if tag:
            g_nid = list(range(max(list(graph.nodes)) + 1))
        g_nlabel = [-1 for i in range(max(g_nid) + 1)]
        g_indeg = [0 for i in range(max(g_nid) + 1)]
        e_u, e_v = [], []
        for u, v in graph.edges:
            e_u.append(u)
            e_u.append(v)
            e_v.append(v)
            e_v.append(u)
        g_edges = [e_u, e_v]
        g_elabel = [graph.edges[u,v]['l'] for u,v in zip(e_u, e_v)]
        g_v_neigh = [[] for i in range(max(g_nid) + 1)]
        for v in graph.nodes:
            g_nlabel[v] = graph.nodes[v]['l']
            g_indeg[v] = graph.degree[v]
            g_v_neigh[v] = list(graph.neighbors(v))
        g_label_dict = defaultdict(list)
        for u, l in graph.nodes('l'):
            if l not in g_label_dict: g_label_dict[l] = []
            g_label_dict[l].append(u)
        return [g_nid,
                g_nlabel,
                g_indeg,
                g_edges,
                g_elabel,
                g_v_neigh,
                g_label_dict]
    @staticmethod
    def nx2qinfo(graph: nx.Graph):
        g_nid = list(graph.nodes)
        g_nlabel = [graph.nodes[i]['l'] for i in graph.nodes]
        g_indeg = [0 for i in graph.nodes] #[graph.degree[i] for i in graph.nodes]
        g_edges = np.array(graph.edges).T.tolist()
        g_elabel = [graph.edges[u,v]['l'] for u,v in graph.edges]
        g_v_neigh = [[] for i in graph.nodes]
        g_label_dict = dict()
        for u, l in graph.nodes('l'):
            if l not in g_label_dict: g_label_dict[l] = []
            g_label_dict[l].append(u)
        for u, v in graph.edges:
            g_v_neigh[u].append(v)
            g_indeg[u] += 1
        return [g_nid,
                g_nlabel,
                g_indeg,
                g_edges,
                g_elabel,
                g_v_neigh,

                g_label_dict]
