from math import log2, ceil
import networkx as nx
from random import shuffle
from tqdm import tqdm
import time
import concurrent
import gc 
from database import *
from helper import *
from emm.util.crypto import *
from emm.des import DESclass
from emm.emm import EMMclass
from hypergraph_helper import hypergraph_to_tree, extract_hypergraph

PURPOSE_HMAC = "hmac"
PURPOSE_ENCRYPT = "encryption"
CHUNK_SIZE = 10000

DES = DESclass()
EMM = EMMclass()
           
class OurHES:
    
    def __init__(self, dataset: str, num_cores: int):
        self.num_cores = num_cores
        self.M1_file = "databases/PathHES-databases/" + dataset + '_M1_file.db'
        self.M2_file = "databases/PathHES-databases/" + dataset + '_M2_file.db'
        self.EM1_file = "databases/PathHES-databases/" + dataset + '_EM1_file.db'
        self.EM2_file = "databases/PathHES-databases/" + dataset + '_EM2_file.db'
        self.key_file = "databases/PathHES-databases/" + dataset + '_key_file.db'
        self.M1_db = sqlite3.connect(self.M1_file)
        self.M2_db = sqlite3.connect(self.M2_file)
        self.EM1_db = sqlite3.connect(self.EM1_file)
        self.EM2_db = sqlite3.connect(self.EM2_file)
        self.key_db = sqlite3.connect(self.key_file)
    

    def key_gen(self, security_parameter: int) -> bytes:
        """
        Given the security parameter, generate keys for DES and EMM and store them in dictionary.
        """
        self.key_db = initialize_database(self.key_file)
        t0 = time.time_ns()  
        self.key_DES = DES.key_gen(security_parameter)
        self.key_EMM = EMM.key_gen(security_parameter)
        t1 = time.time_ns()  
        time_to_gen_key = t1-t0
        write_dict_to_sqlite({b"key_DES": [self.key_DES], b"key_EMM": [self.key_EMM]}, self.key_db)

        return time_to_gen_key
    

    def retrieve_key(self, ) -> bytes:
        """
        Retrieve keys for DES and EMM from key database.
        """
        self.key_DES = list(get_values_for_label(self.key_db, b"key_DES"))[0]
        self.key_EMM = list(get_values_for_label(self.key_db, b"key_EMM"))[0]


    def encrypt_hypergraph(self, H):
        """
        Given a hypergraph H and encrypt H, output plaintext files, encrypted database, and associated benchmarks.
        """
        self.M1_db = initialize_database(self.M1_file)
        self.M2_db = initialize_database(self.M2_file)
        self.EM1_db = initialize_database(self.EM1_file )
        self.EM2_db = initialize_database(self.EM2_file)

        self.H = H
        NUM_NODES = len(H.get_node_set())

        total_num_edges_in_M2 = 0
        t0 = time.time_ns()  
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            for partial_M1, partial_M2, num_edges_in_M2 in tqdm(executor.map(
                    build_mm_helper, ((h, self.H, self.key_EMM) for h in H.get_node_set()), chunksize=20), 
                            total=NUM_NODES, desc="Building multimaps"):
                
                total_num_edges_in_M2 += num_edges_in_M2
                write_dict_to_sqlite(partial_M1, self.M1_db)
                write_dict_to_sqlite(partial_M2, self.M2_db)

        # Pad M1 up to worst case multimap size
        total_num_labels_in_M1 = get_row_count(self.M1_db)   
        num_pad_M1 = NUM_NODES*(NUM_NODES-1) - total_num_labels_in_M1
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            for padded_M1 in tqdm(executor.map(
                   padding_M1_helper, ((chunk, NUM_NODES) for chunk in chunks(num_pad_M1, CHUNK_SIZE))), 
                            total=ceil(num_pad_M1/CHUNK_SIZE), desc="Padding M1"):             
                write_dict_to_sqlite(padded_M1, self.M1_db)
     
        # Pad M2 up to worst case multimap size 4*(NUM_NODES)^2
        num_pad_M2 = 4*(NUM_NODES)**2 - total_num_edges_in_M2
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            for padded_M2 in tqdm(executor.map(
                    padding_M2_helper, chunks(num_pad_M2, CHUNK_SIZE)), 
                            total=ceil(num_pad_M2/CHUNK_SIZE), desc="Padding M2"):
                write_dict_to_sqlite(padded_M2, self.M2_db)

        t1 = time.time_ns()  
        time_to_compute_MMs = t1 - t0

        gc.collect()
        t0 = time.time_ns()    
        DES.build_index(self.M1_db, self.EM1_db, self.num_cores)
        t1 = time.time_ns() 
        gc.collect()
        t2 = time.time_ns() 
        EMM.build_index(self.M2_db, self.EM2_db, self.num_cores)
        t3 = time.time_ns()
        encryption_time = (t1 - t0) + (t3 - t2)
        
        return self.EM1_db, self.EM2_db, self.EM1_file, self.EM2_file, self.M1_file, self.M2_file, time_to_compute_MMs, encryption_time

    
    def compute_token(self, query):
        """
        Given a query, output the corresponding search token.
        """
        query_bytes = str((query[0],query[1])).encode('utf-8')
        return DES.token(self.key_DES, query_bytes)


    def search(self, token):
        """
        Given a search token, output the corresponding encrypted response.
        """
        resp = []
        token_set = DES.search(token, self.EM1_db)
        if token_set == None:
            return []
        
        # Parse tokens 
        token_set = [token_set[i:i+16] for i in range(0, len(token_set), 16)]
        
        # Retrieve each corresponding (encrypted) fragments from EM2  
        for fragment_token in token_set:
            if fragment_token != b"0"*16: # Filter out padding
                fragment = EMM.search(fragment_token, self.EM2_db)
                resp.append(fragment)

        return resp
    

    def reveal(self, resp):
        """
        Given an encrypted response, output the plaintext.
        """
        key_SKE = HashKDF(self.key_EMM, PURPOSE_ENCRYPT)
        pt_path = []
        for ct_fragment in resp:
            pt_fragment = []
            for ct_value in ct_fragment:
                pt_fragment.append(SymmetricDecrypt(key_SKE, ct_value))
            pt_path.append(pt_fragment)
        return pt_path
    

def chunks(max_val, chunk_size):
    """
    A helper function for defining chunks when processing a database.
    """
    for i in range(0, max_val, chunk_size):
        if i + chunk_size <= max_val:
            yield range(i, i + chunk_size)
        else:
            yield range(i, max_val)


def build_mm_helper(params):
    """
    A helper function used to parallelize multimap computation.
    """
    root, H, key_EMM = params
    partial_M1, partial_M2 = {}, {}
    num_edges_in_M2 = 0

    # Compute single destination shortest path tree using the SBT algorithm
    T, root = hypergraph_to_tree(H , root)

    # Compute Heavy-Light decomposition
    size, hl_labels = HLD(T, root, {})
    nx.set_edge_attributes(T, hl_labels,'hl')

    # Compute list of disjoint paths discovered in BFS manner
    Paths = BFS_paths(T, root)

    for path in Paths:
        v, u = path[0], path[-1]
        
        # Pad number of edges in path up to next power of 2, 'p' indicates padding nnode
        power = next_power_of_2(len(path)-1) 
        path = path + ['p']*((2**power)-(len(path)-1))

        for j in range(power+1):
            # Compute fragment of length 2^j edges
            fragment = path[:2**j+1]
                    
            fragment_bytes = []
            for node in fragment:
                if type(node) == int:
                    fragment_bytes.append(str(node).encode('utf-8'))
                elif type(node) == str:
                    fragment_bytes.append(node.encode('utf-8'))

            label_fragment = str((root,u,v,j)).encode('utf-8')
            value_edges = b""
            for index in range(len(fragment_bytes)-1):
                if value_edges == b"":
                    value_edges = fragment_bytes[index]+ b";" + fragment_bytes[index+1]
                    num_edges_in_M2 += 1
                else:
                    value_edges = value_edges + b"/" + fragment_bytes[index]+ b";" + fragment_bytes[index+1]
                    num_edges_in_M2 += 1

            partial_M2[label_fragment] = [value_edges]

            if j == 0:
                subpath = [path[1]]
            else:
                subpath = path[2**(j-1)+1:2**j+1]

            for w in subpath:
                if w != 'p':
                    tk = EMM.token(key_EMM, label_fragment)
                                
                    label_query = str((w,root)).encode('utf-8')
                    prev_query = str((v,root)).encode('utf-8')
                            
                    if prev_query in partial_M1:
                        partial_M1[label_query] = partial_M1[prev_query]+ [tk]
                        shuffle(partial_M1[label_query])
                    else:
                        partial_M1[label_query] = [tk]

    # Pad each list of values in M1 up to log2(len(NUM_NODES))
    pad_len = ceil(log2(len(H.get_node_set())))
    padded_M1 = {label: [b''.join(values + [b"0"*16] * (pad_len - len(values)))] 
            for label, values in partial_M1.items()}
        
    return padded_M1, partial_M2, num_edges_in_M2


def padding_M1_helper(params):
    """
    A helper function used to parallelize padding multimap M1.  
    """
    indices, NUM_NODES = params
    temp = {}
    for i in indices:
        temp[str((i)).encode('utf-8')] = [b''.join([b"0"*16] * ceil(log2(NUM_NODES)))]
    return temp


def padding_M2_helper(indices):
    """
    A helper function used to parallelize padding multimap M2. 
    """
    temp = {}
    for i in indices:
        temp[str((i)).encode('utf-8')] = [b"p"]
    return temp