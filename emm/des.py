from multiprocessing import Pool
import concurrent.futures
from tqdm import tqdm
from .util.crypto import *
from database import *
from math import ceil

PURPOSE_HMAC = "hmac"
PURPOSE_ENCRYPT = "encryption"
CHUNK_SIZE =  1000


class DESclass:

    def __init__(self, ):
        self.key = None
        

    def key_gen(self, security_parameter: int) -> bytes:
        """
        Given security parameter, output secret key.
        """
        self.key = SecureRandom(security_parameter)
        return self.key
    

    def encrypt_helper(self, data_chunk):
        """
        Given a data chunk, output encrypted data in the form of a dictionary.
        """
        temp = {}
        for row in data_chunk:
            label, value = row[0], row[1]
            K1 = HMAC(self.key, b'1'+label)[:16]
            K2 = HMAC(self.key, b'2'+label)[:16]  
            ct_label = Hash(K1)
            ct_value = SymmetricEncrypt(K2, value)
            temp[ct_label] = [ct_value]
        return temp


    def build_index(self, DX_db, EDX_db_conn, num_cores):
        """
        Given database files DX_db and EDX_db, encrypt contents of DX_db and write to EDX_db.
        """
        total_chunks = ceil(get_row_count(DX_db)/CHUNK_SIZE)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        #     for partial_edx in tqdm(executor.map(
        #             self.encrypt_helper, read_data_streaming(DX_db, CHUNK_SIZE),
        #                 ), total=total_chunks, desc="Encrypting with EMM-RR"):
        #         write_dict_to_sqlite(partial_edx, EDX_db_conn)
        for stream in read_data_streaming(DX_db, CHUNK_SIZE):
            partial_edx = self.encrypt_helper(stream)
            write_dict_to_sqlite(partial_edx, EDX_db_conn)


    def token(self, key: bytes, label: bytes) -> bytes:
        """
        Given a label, output the corresponding search token.
        """
        K1 = HMAC(key, b'1'+label)[:16]
        K2 = HMAC(key, b'2'+label)[:16]
        return K1 + K2


    def search(self, search_token, DX_db):
        """
        Given a search token and encrypted database EMM_db, output the corresponding plaintext response.
        """
        #Parse search token
        token_length = int(len(search_token)/2)
        K1, K2 = search_token[:token_length], search_token[token_length:]

        ct_label = Hash(K1)
        ct_values =  list(get_values_for_label(DX_db, ct_label))

        if ct_values != []:
            for ct_value in ct_values:
                return SymmetricDecrypt(K2, ct_value)
        else:
            return None
        

