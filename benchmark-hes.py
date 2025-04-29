import networkx as nx
from helper import *
import secrets
import sys
import time
from tqdm import tqdm
import csv
from hes import OurHES

from halp.directed_hypergraph import DirectedHypergraph
from halp.algorithms.directed_paths import shortest_b_tree

dataset=sys.argv[1]
SETUP_FLAG=int(sys.argv[2])     # If 0 won't run setup experiments.
NUM_QUERIES=int(sys.argv[3])    # If 0 won't run query experiments.
NUM_PROCESSES=int(sys.argv[4])

def generate_random_query(num_nodes):
    v1 = secrets.randbelow(num_nodes)
    v2 = secrets.randbelow(num_nodes)
    while v1 == v2:
        v2 = secrets.randbelow(num_nodes)
    return v1, v2

def run_benchmarks(H):

    setup_results = []
    query_results = []

    if bool(SETUP_FLAG):

        # Measure time to encrypt graph.
        time_to_gen_key = HES.key_gen(16)
        EM1_db, EM2_db, EM1_file, EM2_file, M1_file, M2_file, time_to_compute_MMs, encryption_time = HES.encrypt_graph(H)
        total_setup_time = time_to_gen_key + time_to_compute_MMs + encryption_time

        print("Setup time (ms): ", total_setup_time// 1000000)
        print("Encryption time (ms)", encryption_time// 1000000)
        
        # Measure size of EDB.
        print("Accumulating storage results...")
        EM1_size = os.path.getsize(EM1_file)
        EM2_size = os.path.getsize(EM2_file)
        M1_size = os.path.getsize(M1_file)
        M2_size = os.path.getsize(M2_file)
        
        print("Creating Index on keys...", encryption_time// 1000000)
        cursor = EM1_db.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON byte_data(key)")
        cursor = EM2_db.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON byte_data(key)")

        print("Total EDB size:", EM1_size + EM2_size, "Bytes")

        setup_results.append((EM1_size, EM2_size, M1_size, M2_size, time_to_compute_MMs, encryption_time, total_setup_time))

    if bool(NUM_QUERIES):
        
        HES.retrieve_key()

        print("Running query benchmarks...")
        for _ in tqdm(range(NUM_QUERIES)):
            # source, target = generate_random_query(NUM_NODES)
            source = 7
            target = 1

            mapping, weight_dict = shortest_b_tree(H, target)
            # paths = nx.single_source_shortest_path(H, target)

            # Measure token generation time.
            t0 = time.time_ns()
            tk = HES.compute_token((source, target))
            t1 = time.time_ns()
            token_gen_time = t1 - t0
            
            # Measure search time.
            t0 = time.time_ns()        
            resp = HES.search(tk)
            t1 = time.time_ns()
            search_time = t1 - t0

            # Measure size of response.
            number_of_fragments = len(resp)
            resp_size = 0
            for fragment in resp:
                resp_size += sum((sys.getsizeof(edge) for edge in fragment))

            # Measure time to decrypt path.
            t0 = time.time_ns()
            path = HES.reveal(resp)
            t1 = time.time_ns()
            reveal_time = t1 - t0
            # Compute percent padding.
            total_length = sum((len(fragment) for fragment in path))

            # todo: check if solution is correct.

            # if  source in paths:
            #     true_path = paths[source]
            # else:
            #     true_path = []
            #
            # if path != []:
            #     true_length = len(true_path)-1
            #     total_padding = total_length - true_length
            #     percent_padding = ((total_length-true_length) / total_length) * 100
            # else:
            #     true_length = 0
            #     total_padding = 0
            #     percent_padding = 0
            #
            # plaintext_path_bytes = sys.getsizeof(path)
            #
            # total_query_time = token_gen_time + search_time + reveal_time
            #
            # query_results.append([token_gen_time, search_time, reveal_time, total_query_time,
            #                       true_length, number_of_fragments, total_padding,
            #                       percent_padding, resp_size, plaintext_path_bytes])

            
    return setup_results, query_results

if __name__ == "__main__":
    # todo: Replace with actual datasets.
    H = DirectedHypergraph()
    H.add_hyperedge([1], [2])
    H.add_hyperedge([1], [3])
    H.add_hyperedge([2, 3], [4])
    H.add_hyperedge([1], [5])
    H.add_hyperedge([4, 5], [6])
    H.add_hyperedge([6], [7])


    # ----
    nodes = list(H.node_iterator())

    # G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    # I don't think we need integer labels but just be sure:
    assert all([type(a) is int for a in nodes])

    NUM_NODES = len(nodes)
    HES = OurHES(dataset, NUM_PROCESSES)
    
    setup_results, query_results = run_benchmarks(H)

    if bool(SETUP_FLAG):
        setup_csv = "results/PathGES-Results/PathGES-" + dataset + "-setup.csv"
        setup_fields = ["EM1_size (B)", "EM2_size (B)", "M1_size (B)", "M2_size (B)",
                "time_to_comp_MMs (ns)", "enc_time (ns)", "setup_time (ns)"]

        # Write setup results to csv file.    
        f1 = open(setup_csv, 'a')
        csvwriter1 = csv.writer(f1) 
        csvwriter1.writerow(setup_fields) 
        csvwriter1.writerows(setup_results)
    
    if bool(NUM_QUERIES):
        query_csv = "results/PathGES-Results/PathGES-" + dataset + "-query.csv"
        query_fields = ["token_gen_time (ns)", "search_time (ns)", "reveal_time (ns)", "total_query_time (ns)",
                        "true_length", "number_of_fragments", "total_padding",
                        "percent_padding", "resp_size (B)", "plaintext_path_bytes(B)"]
        
        # Write query results to csv file.
        f2 = open(query_csv, 'w')
        csvwriter2 = csv.writer(f2) 
        csvwriter2.writerow(query_fields) 
        csvwriter2.writerows(query_results)
