from halp.directed_hypergraph import DirectedHypergraph
from halp.algorithms.directed_paths import shortest_b_tree
import networkx as nx
from math import log2, ceil

def extract_hypergraph(dataset):
    """
    Extract a directed hypergraph from a data file.
    Format should be one hyperedge per line with format:
    tail_node1,tail_node2,...:head_node1,head_node2,...
    
    Args:
        dataset (str): Path to the file containing hypergraph data
        
    Returns:
        DirectedHypergraph: The extracted hypergraph
    """
    H = DirectedHypergraph()
    with open("data/" + dataset + ".txt", "r") as hyperedges:
        for line in hyperedges:
            # Skip empty lines or comments
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            parts = line.strip().split(';')
            if len(parts) != 3:
                continue
                
            tail_str, head_str, weight = parts
            tail_nodes = [int(node) for node in tail_str.split(',') if node.strip()]
            head_nodes = [int(node) for node in head_str.split(',') if node.strip()]
            weight = 1 if weight=='' else int(weight)
            
            # Add nodes if they don't exist
            for node in tail_nodes + head_nodes:
                if not H.has_node(node):
                    H.add_node(node)
            
            # Add the hyperedge
            H.add_hyperedge(tail_nodes, head_nodes, weight=weight)
            
    return H

def hypergraph_to_tree(H, id_source_vertex):
    """
    Calculate a tree from a hypergraph using the Shortest B-Tree algorithm.
    
    Parameters:
        H (DirectedHypergraph): The hypergraph
        id_source_vertex (int): The ID of the source vertex
        
    Returns:
        (nx.DiGraph, int): The tree and the root node ID
    """
    # Runs the SBT algorithm for vertex "id_source_vertex" as source.
    mapping, weight_dict = shortest_b_tree(H, id_source_vertex)    

    # Create the tree
    tree = nx.DiGraph()
    tree.add_node(id_source_vertex)

    for vertex_id, hyperedge_id in mapping.items():
        if hyperedge_id is None:
            continue

        tail_vertex_id_list = list(H.get_hyperedge_tail(hyperedge_id))
        tail_vertex_id_list = [t for t in tail_vertex_id_list if t in weight_dict]
        if not tail_vertex_id_list:
            continue

        max_dist = max(weight_dict[t] for t in tail_vertex_id_list)
        maximal_weight_candidates = [t for t in tail_vertex_id_list if weight_dict[t] == max_dist]
        node_id_maximal_weight = maximal_weight_candidates[0] if len(maximal_weight_candidates) == 1 else maximal_weight_candidates[-1]
        tree.add_edge(vertex_id, node_id_maximal_weight)
    
    return tree, id_source_vertex        