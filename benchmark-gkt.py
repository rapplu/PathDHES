import networkx as nx
from helper import *
import secrets
import sys
import time
from tqdm import tqdm
import csv

from gkt import GKT

DATA=sys.argv[1]
NUM_QUERIES=int(sys.argv[2])    # If 0 won't run query experiments.
NUM_PROCESSES=int(sys.argv[3])


def generate_random_query(num_nodes):
    v1 = secrets.randbelow(num_nodes)
    v2 = secrets.randbelow(num_nodes)
    while v1 == v2:
        v2 = secrets.randbelow(num_nodes)
    return v1, v2


def run_benchmarks(G):

    setup_results = []
    query_results = []

    # Measure time to encrypt graph.
    print("Building index...")
    t0 = time.time_ns()
    GES.key_gen(16)
    EDX_db, EDX_file, DX_file, encryption_time = GES.encrypt_graph(G)
    t1 = time.time_ns()
    total_setup_time = t1 - t0
    
    print("Encryption time (ms): ", encryption_time// 1000000)
    print("Setup time (ms): ", total_setup_time// 1000000)

     # Measure size of EDB.
    print("Accumulating storage results...")
    EDX_size = os.path.getsize(EDX_file)
    DX_size = os.path.getsize(DX_file)

    print("Creating Index on keys...")
    cursor = EDX_db.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON byte_data(key)")
    
    setup_results.append((EDX_size, DX_size, encryption_time, total_setup_time))
    
    if bool(NUM_QUERIES):
            
        print("Running query benchmarks...")
        for _ in tqdm(range(NUM_QUERIES)):

            source, target = generate_random_query(NUM_NODES)
            # Measure token generation time.
            t0 = time.time_ns()
            tk = GES.compute_token((source, target))
            t1 = time.time_ns()
            token_gen_time = t1 - t0
            
            # Measure search time.
            t0 = time.time_ns()
            resp = GES.search(tk)
            t1 = time.time_ns()
            search_time = t1 - t0

            # Measure size of response.
            number_of_fragments = len(resp)
            resp_size = sum((sys.getsizeof(item) for item in resp))

            # Measure time to decrypt path.
            t0 = time.time_ns()
            path = GES.reveal(resp)
            t1 = time.time_ns()
            reveal_time = t1 - t0

            # Compute path length and size in Bytes.
            true_length = len(path)
            plaintext_path_bytes = sys.getsizeof(path)
            
            total_query_time = token_gen_time + search_time + reveal_time
            
            query_results.append([token_gen_time, search_time, reveal_time, total_query_time,
                                  true_length, resp_size, plaintext_path_bytes]) 

    
    return setup_results, query_results

if __name__ == "__main__":
    G = extract_graph(DATA)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    NUM_NODES = len(G.nodes())
    GES = GKT(DATA, NUM_PROCESSES)

    setup_csv = "results/GKT-Results/GKT-" + DATA + "-setup.csv"
    setup_fields = ["EDX_size(B)", "DX_size(B)", "encryption_time (ns)", "setup_time (ns)"]

    f1 = open(setup_csv, 'a')
    csvwriter1 = csv.writer(f1) 
    csvwriter1.writerow(setup_fields) 
    
    setup_results, query_results = run_benchmarks(G)

    # Write setup results to csv file.
    csvwriter1.writerows(setup_results)

    if bool(NUM_QUERIES):
        query_csv = "results/GKT-Results/GKT-"+ DATA + "-query.csv"
        query_fields = ["token_gen_time (ns)", "search_time (ns)", "reveal_time (ns)", "total_query_time (ns)",
                        "true_length", "resp_size (B)", "plaintext_path_bytes(B)"]
        
        # Write query results to csv file.
        f2 = open(query_csv, 'w') 
        csvwriter2 = csv.writer(f2) 
        csvwriter2.writerow(query_fields) 
        csvwriter2.writerows(query_results)



    

            
