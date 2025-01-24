from oqs import KeyEncapsulation, Signature
import json
from cryptography.fernet import Fernet
from functools import lru_cache  # Added caching
import concurrent.futures  # Added parallel processing

class QuantumSecureComms:
    def __init__(self):
        self.kem_alg = "Kyber1024"
        self.sig_alg = "Dilithium5"
        self.kem = KeyEncapsulation(self.kem_alg)
        self.sig = Signature(self.sig_alg)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    @lru_cache(maxsize=1000)  # Added caching for keypairs
    def generate_keypair(self):
        """Generate quantum-resistant key pair"""
        public_key = self.kem.generate_keypair()
        signature_key = self.sig.generate_keypair()
        return {
            'kem_pub': public_key,
            'sig_pub': signature_key.public_key,
            'sig_priv': signature_key.private_key
        }

    def encrypt_order(self, order: dict, public_key: bytes) -> bytes:
        """Parallel encryption using quantum-safe KEM"""
        future = self.executor.submit(self._encrypt_data, order, public_key)
        return future.result()

    def _encrypt_data(self, order: dict, public_key: bytes) -> bytes:
        """Internal encryption method"""
        ciphertext, shared_secret = self.kem.encap_secret(public_key)
        f = Fernet(shared_secret)
        encrypted_data = f.encrypt(json.dumps(order).encode())
        return ciphertext + encrypted_data

    def verify_signed_data(self, data: bytes, sig: bytes, public_key: bytes) -> bool:
        """Optimized signature verification"""
        return self.sig.verify(data, sig, public_key)
