class KellyCriterionOptimizer:
    def __init__(self):
        self.win_rate = 0.55  # Continually updated
        self.loss_rate = 0.45
        self.avg_win = 1.2    # Average win/loss ratio
        self.avg_loss = 1.0
        
    def optimal_position_size(self):
        return (self.win_rate / self.avg_loss) - (self.loss_rate / self.avg_win)
