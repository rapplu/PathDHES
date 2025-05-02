from helper import *
from hypergraph_helper import *
import sys
from hes import OurHES

dataset=sys.argv[1]
SOURCE=int(sys.argv[2])
TARGET=int(sys.argv[3])   
SETUP_FLAG=int(sys.argv[4]) 
NUM_PROCESSES=int(sys.argv[5])

def run_dp_query(H, query):
    source, target = query

    if source not in H.get_node_set() or target not in H.get_node_set():
        print("Source or target node not in hypergraph. Query cannot be run.")
        return None
    
    if SETUP_FLAG:
        HES.key_gen(16)
        HES.encrypt_hypergraph(H)

    HES.retrieve_key()

    print("Running requested query...")
    tk = HES.compute_token((target, source))  
    resp = HES.search(tk)
    direct_path = HES.reveal(resp)
    print(direct_path)
    return direct_path


if __name__ == "__main__":
    H = extract_hypergraph(dataset)
    nodes = list(H.node_iterator())

    assert all([type(a) is int for a in nodes])

    NUM_NODES = len(nodes)
    HES = OurHES(dataset, NUM_PROCESSES)
    
    direct_path = run_dp_query(H, (SOURCE, TARGET))