import os
from cryptography.fernet import Fernet

class CredentialManager:
    def __init__(self):
        self.key = os.getenv("CRYPTO_KEY")
        self.cipher = Fernet(self.key)
        
    def decrypt_credentials(self, encrypted_file):
        with open(encrypted_file, 'rb') as f:
            return self.cipher.decrypt(f.read()).decode()
