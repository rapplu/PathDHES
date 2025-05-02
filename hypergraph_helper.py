from halp.directed_hypergraph import DirectedHypergraph
from halp.algorithms.directed_paths import shortest_b_tree
import networkx as nx


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


def get_nodes(H):
    nodes = list(H.node_iterator())
    nodes.sort()

    num_nodes = len(nodes)
    num_nodes_plues_helper = 0

    for node in nodes:
        tail_size_list = [len(H.get_hyperedge_tail(e)) for e in H.get_backward_star(node)]
        num_nodes_plues_helper += 1
        if len(tail_size_list) > 0 and max(tail_size_list) > 1:
            num_nodes_plues_helper += max(tail_size_list)
    
    return num_nodes, num_nodes_plues_helper


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


def extract_b_path_trees(H, root):
    '''
        Perform the B-Path encryption preprocessing trick.
        Extract all the necessart trees from the hypergraph.
    '''
    # Runs the SBT algorithm for root as source.
    mapping, weight_dict = shortest_b_tree(H, root)

    weight_tuples = [(vertex_id, weight_dict[vertex_id]) for vertex_id in weight_dict.keys()]
    weight_tuples.sort(key=lambda x: x[1])

    # For each node that was already analyzed, the mapping returns the tree in which the node is part of
    # (A node can be the leaf in multiple trees but only has outgoing connections in one tree. The mapping returns this tree.)
    node_tree_mapping = {}

    main_tree = nx.DiGraph()
    main_tree.add_node(str(root))
    node_tree_mapping[str(root)] = main_tree

    root_tree_list = []
    root_tree_list.append((str(root), main_tree))

    root_nodes_num_pointers = {}

    for (vertex_id, _) in weight_tuples:
        hyperedge_id = mapping[vertex_id]
        if hyperedge_id is None:
            continue

        tail_vertex_id_list = H.get_hyperedge_tail(hyperedge_id)

        # todo: think about this case again, probably cannot happen anyway.
        if len(tail_vertex_id_list) == 0:
            continue

        if len(tail_vertex_id_list) == 1:
            tail_vertex_id = tail_vertex_id_list[0]
            tree_of_tail_vertex = node_tree_mapping[str(tail_vertex_id)]

            tree_of_tail_vertex.add_edge(str(vertex_id), str(tail_vertex_id))  #
            node_tree_mapping[str(vertex_id)] = tree_of_tail_vertex
        else:
            # Tail has multiple vertices: Start new tree.
            new_tree = nx.DiGraph()
            node_tree_mapping[str(vertex_id)] = new_tree
            root_tree_list.append((str(vertex_id), new_tree))

            for idx, tail_vertex_id in enumerate(tail_vertex_id_list):
                tree_of_tail_vertex = node_tree_mapping[str(tail_vertex_id)]
                # Todo: ensure that no tail has less than 100 vertices.
                tree_of_tail_vertex.add_edge(str(vertex_id) + "_" + str(idx), str(tail_vertex_id))  # v

            root_nodes_num_pointers[str(vertex_id)] = len(tail_vertex_id_list)

    return root_tree_list, root_nodes_num_pointers