import itertools
import os

import numpy as np


class AwesomeGridSearch(object):
    def __init__(self, net_constr, X, y):
        self.net_constr = net_constr

        # TODO: Add rand data init
        self.X = X
        self.y = y
        self.err_hist = []

    def search(self, params):
        keys,vals = tuple(params.keys()), tuple(params.values())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        for experiment in experiments:
            print('Trying: {}'.format(experiment))
            net = self.net_constr(**experiment)
            net.fit(self.X, self.y)
            net.train()
            self.err_hist.append(np.average(net.test_err))
            print('Score: {}{}'.format(self.err_hist[-1], os.linesep))

        best_ind = np.argmax(self.err_hist)
        worst_ind = np.argmin(self.err_hist)
        best_params = experiments[best_ind]
        worst_params = experiments[worst_ind]
        best_err = self.err_hist[best_ind]
        worst_err = self.err_hist[worst_ind]

        return best_params, best_err, worst_params, worst_err