import networkx as nx
import numpy as np
import os
from copy import deepcopy
from collections import defaultdict

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

def load_g_graph(g_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    elabel = list()
    e_u = list()
    e_v = list()
    with open(g_file) as f2:
        num_nodes = int(f2.readline().rstrip())
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
        for i in range(num_nodes):
            node_info = f2.readline()
            node_id, node_label = node_info.rstrip().split()
            nid.append(int(node_id))
            nlabel.append(int(node_label))
        while True:
            line = f2.readline()
            if not line:
                break
            temp_indeg = int(line.strip())
            nindeg.append(temp_indeg)
            if temp_indeg == 0:
                continue
            for i in range(temp_indeg):
                edge_info = f2.readline().rstrip().split()
                if len(edge_info) == 2:
                    edge_label = 1
                else:
                    edge_label = int(edge_info[-1])
                e_u.append(int(edge_info[0]))
                e_v.append(int(edge_info[1]))
                v_neigh[int(edge_info[0])].append(int(edge_info[1]))
                elabel.append(edge_label)
    g_nid = deepcopy(nid)
    g_nlabel = deepcopy(nlabel)
    g_indeg = deepcopy(nindeg)
    g_edges = [deepcopy(e_u), deepcopy(e_v)]
    g_elabel = deepcopy(elabel)
    g_v_neigh = deepcopy(v_neigh)
    g_label_dict = defaultdict(list)
    for i in range(len(g_nlabel)):
        g_label_dict[g_nlabel[i]].append(i)
    graph_info = [
        g_nid,
        g_nlabel,
        g_indeg,
        g_edges,
        g_elabel,
        g_v_neigh,
        g_label_dict
    ]
    return graph_info


def load_p_data(p_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    elabel = list()
    e_u = list()
    e_v = list()

    with open(p_file) as f1:
        num_nodes = int(f1.readline().rstrip())
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
        for i in range(num_nodes):
            node_info = f1.readline()
            node_id, node_label = node_info.rstrip().split()
            nid.append(int(node_id))
            nlabel.append(int(node_label))
        while True:
            line = f1.readline()
            if not line:
                break
            temp_indeg = int(line.strip())
            nindeg.append(temp_indeg)
            if temp_indeg == 0:
                continue
            for i in range(temp_indeg):
                edge_info = f1.readline().rstrip().split()
                if len(edge_info) == 2:
                    edge_label = 1
                else:
                    edge_label = int(edge_info[-1])
                e_u.append(int(edge_info[0]))
                e_v.append(int(edge_info[1]))
                v_neigh[int(edge_info[0])].append(int(edge_info[1]))
                #v_neigh[int(edge_info[1])].append(int(edge_info[0]))
                elabel.append(edge_label)
    p_nid = deepcopy(nid)
    p_nlabel = deepcopy(nlabel)
    p_indeg = deepcopy(nindeg)
    p_edges = [deepcopy(e_u), deepcopy(e_v)]
    p_elabel = deepcopy(elabel)
    p_v_neigh = [deepcopy(v_list) for v_list in v_neigh]
    p_label_dict = defaultdict(list)
    for i in range(len(p_nlabel)):
        p_label_dict[p_nlabel[i]].append(i)
    pattern_info = [
        p_nid,
        p_nlabel,
        p_indeg,
        p_edges,
        p_elabel,
        p_v_neigh,
        p_label_dict
    ]
    return pattern_info


class QDecomposer():
    def __init__(self):
        pass
    def decompose(self, query: nx.Graph, **para):
        raise NotImplementedError

class GDecomposer():
    def __init__(self):
        pass
    def decompose(self, graph: nx.Graph, query:nx.Graph, **para):
        raise NotImplementedError

class RdmQDecomposer(QDecomposer):
    def __init__(self):
        super().__init__()
    def decompose(self, query: nx.Graph, **para):
        k = para['k'] if 'k' in para else 12
        k = len(query.nodes) if len(query.nodes) < k else k
        res = []
        notselected = set(query.nodes)
        ug = query.to_undirected()
        while notselected:
            thisturm = [np.random.choice(list(notselected))]
            for _ in range(k - 1):
                neibors = set()
                for v in thisturm: neibors = neibors.union(set(query.neighbors(v)))
                neibors -= set(thisturm)
                thisturm.append(np.random.choice(list(neibors)))
            res.append(thisturm)
            notselected -= set(thisturm)
        return res

class NoQDecomposer(QDecomposer):
    def __init__(self):
        super().__init__()
    def decompose(self, query: nx.Graph, **para):
        return [list(query.nodes), list(query.nodes)]

from filtering import Filtering
from preprocess import SampleSubgraph

class GQLGDecomposer(GDecomposer):
    def __init__(self, name):
        super().__init__()
        self.dataname = name
        self.graph_data = load_g_graph('../data/%s/data.grf' % name)
        self.filter_model = Filtering(None, self.graph_data)
        self.subgraph_sampler = SampleSubgraph(None, self.graph_data)
    def decompose(self, graph: nx.Graph, query: nx.Graph, **para):
        assert 'query_name' in para
        self.filter_model.update_query(load_p_data(para['query_name']))
        self.subgraph_sampler.update_query(load_p_data(para['query_name']))
        candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = self.filter_model.cpp_GQL(os.path.abspath(para['query_name'].replace('.grf', '.graph')), os.path.abspath('../data/%s/data2.graph' % self.dataname))
        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = self.subgraph_sampler.load_induced_subgraph(candidates,induced_subgraph_list, neighbor_offset)
        out_vertices, out_v_label, out_degree, out_edges, out_e_label, out_v_neigh = [], [], [], [], [], []
        substructures = []
        for i in range(len(new_vertices)):
            if len(new_vertices[i]) <= 1: continue
            out_vertices.append(new_vertices[i]) 
            out_v_label.append(new_v_label[i]) 
            out_degree.append(new_degree[i])
            out_edges.append(new_edges[i])
            out_e_label.append(new_e_label[i])
            out_v_neigh.append(new_v_neigh[i])
            substructures.append(constructnx(new_vertices[i], new_v_label[i], new_edges[i], 0))
        return substructures, candidate_info
        
class GQLGDecomposerN(GDecomposer):
    def __init__(self, name):
        super().__init__()
        self.dataname = name
        self.graph_data = load_g_graph('../data/%s/data.grf' % name)
        self.filter_model = Filtering(None, self.graph_data)
        self.subgraph_sampler = SampleSubgraph(None, self.graph_data)
    def decompose(self, graph: nx.Graph, query: nx.Graph, **para):
        assert 'query_name' in para
        self.filter_model.update_query(load_p_data(para['query_name']))
        self.subgraph_sampler.update_query(load_p_data(para['query_name']))
        candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = self.filter_model.cpp_GQL(para['query_name'].replace('.grf', '.graph'), '../data/%s/data2.graph' % self.dataname)
        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = self.subgraph_sampler.find_subgraph_reduced(candidate_info, query, graph)
        out_vertices, out_v_label, out_degree, out_edges, out_e_label, out_v_neigh = [], [], [], [], [], []
        substructures = []
        for i in range(len(new_vertices)):
            if len(new_vertices[i]) <= 1: continue
            out_vertices.append(new_vertices[i]) 
            out_v_label.append(new_v_label[i]) 
            out_degree.append(new_degree[i])
            out_edges.append(new_edges[i])
            out_e_label.append(new_e_label[i])
            out_v_neigh.append(new_v_neigh[i])
            substructures.append(constructnx(new_vertices[i], new_v_label[i], new_edges[i], 0))
        return substructures, candidate_info
        
