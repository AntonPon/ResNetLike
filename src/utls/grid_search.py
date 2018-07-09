import itertools

import numpy as np


class AwesomeGridSearch(object):
    def __init__(self, net_constr, X, y):
        self.net_constr = net_constr

        # TODO: Add rand data init
        self.X = X
        self.y = y
        self.err_hist = []

    def search(self, params):
        keys,vals = zip(*params)
        experiments = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        for experiment in experiments:
            net = self.net_constr(**experiment)
            net.fit(self.X, self.y)
            net.train()
            self.err_hist.append(np.average(net.test_err))

        best_ind = np.argmin(self.err_hist)
        worst_ind = np.argmax(self.err_hist)
        best_params = params[best_ind]
        worst_params = params[worst_ind]
        best_err = params[best_ind]
        worst_err = params[worst_ind]

        return best_params, best_err, worst_params, worst_err