import cython
from typing import Dict, Any
from ..security.oqs_comms import QuantumSecureComms
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import deque
import logging

class TradingEngine:
    def __init__(self):
        self.quantum_security = QuantumSecureComms()
        self.keypair = self.quantum_security.generate_keypair()
        # Added optimized thread pool with CPU core-aware sizing
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4))
        # Increased queue size and using deque for faster operations
        self.order_queue = asyncio.Queue(maxsize=5000)
        self.order_cache = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        # Added performance monitoring
        self.execution_times = deque(maxlen=100)
        self.last_optimization = time.time()

    @cython.ccall
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def execute_order(self, order: cython.double) -> cython.bint:
        try:
            start_time = time.time()
            
            # Batch processing for multiple orders
            if len(self.order_cache) >= 10:
                orders = list(self.order_cache)
                self.order_cache.clear()
                encrypted_orders = self.quantum_security.batch_encrypt_orders(
                    orders, 
                    self.keypair['kem_pub']
                )
            else:
                encrypted_order = self.quantum_security.encrypt_order(
                    order, 
                    self.keypair['kem_pub']
                )
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Auto-optimization if performance degrades
            if len(self.execution_times) >= 50:
                avg_time = np.mean(self.execution_times)
                if avg_time > 0.075:  # 75ms threshold
                    self._optimize_performance()
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return False

    async def process_orders(self):
        """Optimized async order processing with batching"""
        batch = []
        while True:
            try:
                order = await asyncio.wait_for(self.order_queue.get(), timeout=0.01)
                batch.append(order)
                
                if len(batch) >= 10:
                    await asyncio.gather(*[
                        asyncio.get_event_loop().run_in_executor(
                            self.thread_pool,
                            self.execute_order,
                            order
                        ) for order in batch
                    ])
                    batch = []
                    
            except asyncio.TimeoutError:
                if batch:
                    await asyncio.gather(*[
                        asyncio.get_event_loop().run_in_executor(
                            self.thread_pool,
                            self.execute_order,
                            order
                        ) for order in batch
                    ])
                    batch = []

    def _optimize_performance(self):
        """Dynamic performance optimization"""
        current_time = time.time()
        if current_time - self.last_optimization < 300:  # 5 minutes cooldown
            return
            
        self.last_optimization = current_time
        self.thread_pool._max_workers = min(
            64,
            self.thread_pool._max_workers + 4
        )
        self.execution_times.clear()
