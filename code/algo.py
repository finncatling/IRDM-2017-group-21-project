import numpy as np


class PRank:
    def __init__(self, input_dim, k):
        self.k = k
        self.w = np.zeros(input_dim)
        self.b = np.zeros(k)
        self.b[-1] = 99999999999999999999
        pass

    def train(self, x, true_labels):
        for t in range(x.shape[0]):
            x_t = x[t, :]
            y_pred_t = self.predict(x_t)
            y_t = true_labels[t]
            if y_pred_t != y_t:
                tau = np.zeros(self.k - 1)
                for r in range(self.k - 1):
                    if y_t <= (r+1):
                        y_r = -1
                    else:
                        y_r = 1
                    if (np.dot(self.w, x_t)-self.b[r])*y_r <=0:
                        tau[r] = y_r
                    else:
                        tau[r] = 0
                # Update step
                self.w += np.sum(tau)*x_t
                self.b[:-1] -= tau
        pass

    def predict(self, x):
        y_pred = self.k
        for i in range(self.k):
            r =  np.dot(self.w, x) - self.b[i]
            if r < 0:
                y_pred = i+1
                break
        return y_pred