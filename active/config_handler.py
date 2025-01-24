# Replace existing encryption methods with:
def encrypt_log_entry(self, log_data: dict) -> bytes:
    """FIPS 140-3 compliant encryption"""
    iv = os.urandom(16)
    cipher = Cipher(
        algorithms.AES(self.log_key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(json.dumps(log_data).encode()) + encryptor.finalize()
    return iv + encryptor.tag + encrypted

def decrypt_log_entry(self, encrypted_data: bytes) -> dict:
    """Validates integrity during decryption"""
    iv = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = Cipher(
        algorithms.AES(self.log_key),
        modes.GCM(iv, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    return json.loads(decryptor.update(ciphertext) + decryptor.finalize())
