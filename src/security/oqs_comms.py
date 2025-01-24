from oqs import KeyEncapsulation, Signature
import json

class QuantumSecureComms:
    def __init__(self):
        self.kem_alg = "Kyber1024"
        self.sig_alg = "Dilithium5"
        self.kem = KeyEncapsulation(self.kem_alg)
        self.sig = Signature(self.sig_alg)
        
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
        """Encrypt trade order using quantum-safe KEM"""
        ciphertext, shared_secret = self.kem.encap_secret(public_key)
        # Use shared secret with AES-GCM
        encrypted_data = aes_encrypt(json.dumps(order), shared_secret)
        return ciphertext + encrypted_data

    def verify_signed_data(self, data: bytes, sig: bytes, public_key: bytes) -> bool:
        """Verify quantum-safe signature"""
        return self.sig.verify(data, sig, public_key)
