from math import log2, ceil
from multiprocessing import Pool
import networkx as nx
from tqdm import tqdm
import time
import concurrent.futures

from database import *
from helper import *
from emm.util.crypto import *
from emm.des import DESclass

DES = DESclass()
CHUNK_SIZE = 100

class GKT:
    
    def __init__(self, dataset: str, num_cores):
        self.num_cores = num_cores
        self.DX_file = "databases/GKT-databases/" + dataset + '_DX_file.db'
        self.EDX_file = "databases/GKT-databases/" + dataset + '_EDX_file.db'
        self.DX_db = initialize_database(self.DX_file)
        self.EDX_db = initialize_database(self.EDX_file)


    def key_gen(self, security_parameter):
        """
        Given the security parameter, generate keys for SKE and DES.
        """
        self.key_SKE = SecureRandom(security_parameter)
        self.key_DES = DES.key_gen(security_parameter)

    
    def encrypt_graph(self, G):
        """
        Given a graph G, encrypt G and output plaintext file, encrypted database, and associated benchmarks.
        """
        self.G = G
        self.NUM_NODES = len(self.G)

        #Build index
        self.computeSPDX()

        #Encrypt database
        t0 = time.time_ns()
        DES.build_index(self.DX_db, self.EDX_db, self.num_cores)
        t1 = time.time_ns()
        encryption_time = t1 - t0   
        return self.EDX_db, self.EDX_file, self.DX_file, encryption_time


    def compute_token(self, query):
        """
        Given a query, output the corresponding search token.
        """
        label = int_to_bytes(query[0]) + ";".encode('utf-8') + int_to_bytes(query[1])
        return DES.token(self.key_DES,label)


    def search(self, token):
        """
        Given a search token, output the corresponding encrypted response in self.encrypted_db.
        """
        K1 = token[:16]
        curr = token
        resp = []
        value = False
        
        while(True):
            value = DES.search(curr, self.EDX_db)
            
            if value == None:
                return resp
            
            curr = value[:32]
            K1 = value[:16]
            resp.append(value[32:])


    def reveal(self, resp):
        """
        Given an encrypted response, outputs the plaintext path.
        """
        path = []
        for enc_node in resp:
            node = SymmetricDecrypt(self.key_SKE, enc_node)
            node = int(str(node).split(';', 1)[0][2:])
            path.append(node)
        return path


    def computeSPDX(self):
        """
        Outputs a dictionary SPDX containing all of the shortest paths.
        """  
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for partial_SPDX in tqdm(executor.map(
                computeSDSP, ((v, self.G, self.key_SKE, self.key_DES) for v in self.G.nodes()), chunksize=CHUNK_SIZE), 
                            total=self.NUM_NODES, desc="Computing DX"):
                write_dict_to_sqlite(partial_SPDX, self.DX_db)


def computeSDSP(params):
    """
    Takes as input a graph G, a root, and keys, and outputs a multimap containing the query to encrypted value.
    """
    root, G, key_SKE, key_DES = params 
    paths = nx.single_source_shortest_path(G, root)
        
    S = set()
    for start, path in paths.items():   
        path.reverse()
        if len(path)>1:
            for i in range(len(path)-1):
                label = (path[i], root)
                value = (path[i+1],root)
                S.add((label, value))
    M = {}
    for pair in S:
        label, value = pair[0], pair[1]
        label_bytes = int_to_bytes(label[0]) + ";".encode('utf-8') + int_to_bytes(label[1])
        value_bytes = int_to_bytes(value[0]) + ";".encode('utf-8') + int_to_bytes(value[1])
        
        token = DES.token(key_DES, value_bytes)
        ct = SymmetricEncrypt(key_SKE,value_bytes)
        ct_value = token + ct
        if label_bytes not in M:
            M[label_bytes] = [ct_value]        
    return M
