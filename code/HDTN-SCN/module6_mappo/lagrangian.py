# Module 6 — dual-ascent Lagrangian on the capacity/overload constraint (safe RL).
class LagrangianDual:
    def __init__(self, init_beta=0.0, lr=0.05, limit=0.0, beta_max=50.0):
        self.beta = init_beta
        self.lr = lr
        self.limit = limit
        self.beta_max = beta_max

    def update(self, mean_violation):
        self.beta += self.lr * (mean_violation - self.limit)
        self.beta = max(0.0, min(self.beta_max, self.beta))
        return self.beta

    def value(self):
        return self.beta
