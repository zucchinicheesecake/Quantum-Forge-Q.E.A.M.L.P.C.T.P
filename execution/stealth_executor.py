from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class OrderSlice:
    size: float
    price: float
    timestamp: float

class StealthExecutor:
    def __init__(self, exchange_adapter):
        self.exchange = exchange_adapter
        self.min_slice_size = 0.01
        
    def fragment_order(self, total_size: float, urgency: float) -> List[OrderSlice]:
        n_slices = int(total_size / self.min_slice_size)
        times = self._generate_execution_times(n_slices, urgency)
        
        slices = []
        remaining_size = total_size
        
        for t in times:
            slice_size = self._calculate_slice_size(remaining_size, urgency)
            slices.append(OrderSlice(size=slice_size, timestamp=t))
            remaining_size -= slice_size
            
        return slices
        
    def _generate_execution_times(self, n_slices: int, urgency: float) -> List[float]:
        # Poisson arrival process
        return np.random.exponential(1/urgency, n_slices)
