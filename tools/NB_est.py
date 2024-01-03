import numpy as np
from scipy.special import gammaln, factorial
from scipy.optimize import fmin_l_bfgs_b as optim

import tools.util as ut


def fit_nbinom(X, initial_params=None):
    infinitesimal = np.finfo(float).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        # MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = np.sum(gammaln(X + r)) \
            - np.sum(np.log(factorial(X))) \
            - N*(gammaln(r)) \
            + N*r*np.log(p) \
            + np.sum(X*np.log(1-(p if p < 1 else 1-infinitesimal)))

        return -result

    if initial_params is None:
        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m**2)/(v-m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size+m) if size+m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    return {'size': params[0], 'prob': params[1]}


def negbin_mean_to_numpy(mu, b):
    r = b
    var = mu + (1 / b) * mu ** 2
    p = (var - mu) / var

    return r, 1 - p


def negbin_numpy_to_mean(r, p):
    b = r
    mu = r * (1 - p) / p
    return mu, b


def estimate_overdisp_nb(adata, layer=None, cutoff=0.01):

    count_data = ut.convert_to_dense_counts(adata, layer)
    n, p = count_data.shape

    overdisps = []
    means = []

    c = 0
    for i in range(p):
        c += 1
        if c % 100 == 0:
            print(f"Fitting feature {c}/{p}")
        res_ = fit_nbinom(count_data[:, i])
        mu_, b_ = negbin_numpy_to_mean(res_["size"], res_["prob"])

        means.append(mu_)
        overdisps.append(b_)

    adata.var["nb_mean"] = means
    adata.var["nb_overdisp"] = overdisps

    adata.var["nb_overdisp_cutoff"] = adata.var["nb_overdisp"]
    adata.var["nb_overdisp_cutoff"][adata.var["nb_overdisp_cutoff"] < cutoff] = cutoff
    adata.var["nb_overdisp_cutoff"][(adata.var["nb_overdisp"] > 0.1 * adata.var["total_counts"])] = cutoff
