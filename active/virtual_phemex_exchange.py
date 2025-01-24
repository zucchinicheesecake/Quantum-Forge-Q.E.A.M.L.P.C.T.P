# Add these methods to the VirtualPhemexExchange class
def _save_state(self):
    """Atomic write with write-through caching"""
    temp_file = "testnet_state.tmp"
    with open(temp_file, "wb") as f:
        state_data = {
            "balance": self.mUSD_balance,
            "positions": self.positions,
            "checksum": hashlib.sha256(str(self.mUSD_balance).encode()).hexdigest()
        }
        encrypted = self.config.cipher.encrypt(pickle.dumps(state_data))
        f.write(encrypted)
    os.replace(temp_file, "testnet_state.bin")  # Atomic replacement

def _load_state(self):
    if os.path.exists("testnet_state.bin"):
        with open("testnet_state.bin", "rb") as f:
            decrypted = pickle.loads(self.config.cipher.decrypt(f.read()))
            if decrypted["checksum"] == hashlib.sha256(str(decrypted["balance"]).encode()).hexdigest():
                self.mUSD_balance = decrypted["balance"]
                self.positions = decrypted["positions"]
