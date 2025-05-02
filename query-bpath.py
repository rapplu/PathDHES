from helper import *
from hypergraph_helper import *
import sys
from b_path_hes import OurBPathHES

dataset=sys.argv[1]
SOURCE=int(sys.argv[2])
TARGET=int(sys.argv[3])   
SETUP_FLAG=int(sys.argv[4])
NUM_PROCESSES=int(sys.argv[5])

def run_bp_query(H, query):
    source, target = query

    if source not in H.get_node_set() or target not in H.get_node_set():
        print("Source or target node not in hypergraph. Query cannot be run.")
        return None

    if bool(SETUP_FLAG):
        BPathHES.key_gen(16)
        BPathHES.encrypt_hypergraph(H)
    
    BPathHES.retrieve_key()

    print("Running requested query...")

    tk = BPathHES.compute_token((target, source))
    resp = BPathHES.search(tk)
    b_path = BPathHES.reveal(resp)
    print(b_path)
    return b_path

if __name__ == "__main__":
    H = extract_hypergraph(dataset)
    nodes = list(H.node_iterator())

    assert all([type(a) is int for a in nodes])

    NUM_NODES = len(nodes)
    BPathHES = OurBPathHES(dataset, NUM_PROCESSES)
    
    b_path = run_bp_query(H, (SOURCE, TARGET))