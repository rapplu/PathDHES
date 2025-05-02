from typing import *
import multiprocessing.pool as mpp
from multiprocessing import Pool
import networkx as nx
import sys
from emm.util.crypto import *


def chunk_range(max_val, chunk_size):
    for i in range(0, max_val, chunk_size):
        if i + chunk_size <= max_val:
            yield range(i, i + chunk_size)
        else:
            yield range(i, max_val)


def int_to_bytes(x: int) -> bytes:
    str_val = str(x)
    return str_val.encode()


def encode_val(val):
    if type(val) == int:
        return str(val).encode('utf-8')
    elif type(val) == str:
        return val.encode('utf-8')
    

def calc_value_edges(fragment_bytes, num_edges_in_M2):
    value_edges = b""
    for index in range(len(fragment_bytes) - 1):
        if value_edges == b"":
            value_edges = fragment_bytes[index] + b";" + fragment_bytes[index + 1]
            num_edges_in_M2 += 1
        else:
            value_edges = value_edges + b"/" + fragment_bytes[index] + b";" + fragment_bytes[index + 1]
            num_edges_in_M2 += 1

    return value_edges, num_edges_in_M2


def next_power_of_2(x):  
    return 0 if x == 0 else (x - 1).bit_length()


def extract_graph(data):
    G = nx.Graph()
    with open("data/" + data + ".txt", "r") as edges:

        for line in edges:
            p = line.split()
            e = (int(p[0]), int(p[1]))
            G.add_edge(*e)
            
    return G


def SDSP_tree(G, root):
    paths = nx.single_source_shortest_path(G, root)
    T = nx.DiGraph()
    T.add_node(root)

    for u in paths:             
        if len(paths[u])>1:
            for i in range(len(paths[u])-1):
                e = (paths[u][i+1], paths[u][i])
                T.add_edge(*e)

    return T, root


def HLD(T, v, edge_labels):
    sys.setrecursionlimit(8000)
    
    #If node is a leaf, then it has subtree size 1
    if T.out_degree(v)==1 and T.in_degree(v)==0:
        return 1, edge_labels
    
    else:
        v_size = 0
        temp = {}

        #Compute size of subtree rooted at v
        for w in T.predecessors(v):
            w_size, edge_labels = HLD(T, w, edge_labels)
            temp[w] = w_size
            v_size = v_size + w_size
        v_size = v_size + 1
        
        for w, w_size in temp.items():
            #Determine if (w,v) is light or heavy
            if w_size < v_size/2:
                edge_labels[(w,v)] = 'light'
            else:
                edge_labels[(w,v)] = 'heavy'
        
        return v_size, edge_labels

    
def BFS_paths(T, root):
    
    Paths = []
    visited = set()
    queue = []
    visited.add(root)
    
    for w in T.predecessors(root):
        visited.add(w)
        queue.append((root,w))
    
    while queue:
        pair = queue.pop(0)
        v, child = pair[0], pair[1]
        path = [v, child]

        more_path = True
        while more_path:
            prev_node = path[-1]
            end = 0
            for u in T.predecessors(prev_node):
                if T[u][prev_node]['hl'] == 'heavy':
                    visited.add(u)
                    path.append(u)
                    end = 1
                else:
                    visited.add(u)
                    queue.append((prev_node, u))
                        
            if end == 0:
                Paths.append(path)
                more_path = False
        
    return Paths