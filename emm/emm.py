from multiprocessing import Pool
from tqdm import tqdm
import concurrent
from math import ceil

from .util.crypto import *
from database import *

PURPOSE_HMAC = "hmac"
PURPOSE_ENCRYPT = "encryption"
CHUNK_SIZE = 1000

class EMMclass:

    def __init__(self, ):
        self.key = None 


    def key_gen(self, security_parameter):
        """
        Given security parameter, output secret key.
        """
        self.key = SecureRandom(security_parameter)
        self.key_HMAC = HashKDF(self.key, PURPOSE_HMAC)
        self.key_SKE = HashKDF(self.key, PURPOSE_ENCRYPT)
        return self.key
    
            
    def encrypt_data(self, data_chunk):
        """
        Given a data chunk, output encrypted data in the form of a dictionary.
        """
        temp_dict = {}
        for row in data_chunk:
            label, values = row[0], row[1]
            values = values.split(b"/")
            
            token = HMAC(self.key_HMAC, label)[:16]
            ctr = 0
            for value in values:
                ct_label = Hash(token + str(ctr).encode('UTF-8'))
                ct_value = SymmetricEncrypt(self.key_SKE, value)
                temp_dict[ct_label] = [ct_value]
                ctr += 1 
        return temp_dict


    def build_index(self, MM_db, EMM_db, num_cores):
        """
        Given database files MM_db and EMM_db, encrypt contents of MM_db and write to EMM_db.
        """
        total_chunks = ceil(get_row_count(MM_db)/CHUNK_SIZE)
        # Read data from source database in chunks and encrypt
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            for partial_emm in tqdm(executor.map(
                    self.encrypt_data, read_data_streaming(MM_db, CHUNK_SIZE)), total=total_chunks, desc="Encrypting with EMM-RH"):
                write_dict_to_sqlite(partial_emm, EMM_db)


    def token(self, key, label):
        """
        Given a label, output the corresponding search token.
        """
        key_HMAC = HashKDF(key, PURPOSE_HMAC)
        return HMAC(key_HMAC, label)[:16]


    def search(self, search_token, EMM_db):
        """
        Given a search token and encrypted database EMM_db, output the corresponding encrypted response.
        """
        results = []
        # Iterate until can't find any more records:
        ctr = 0
        while True:
            ct_label = Hash(search_token + str(ctr).encode('UTF-8'))
            ct_values = list(get_values_for_label(EMM_db, ct_label))
            if ct_values != []:
                results += ct_values
            else:
                break
            ctr += 1
        return results


    def reveal(self, key, results):
        """
        Given an encrypted response, output the plaintext values.
        """
        key_SKE = HashKDF(key, PURPOSE_ENCRYPT)
        pt_values = []
        for ct_value in results:
            pt_values.append(SymmetricDecrypt(key_SKE, ct_value))
        return pt_values



