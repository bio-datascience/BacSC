import numpy as np
from scipy.special import gammaln, factorial
from scipy.optimize import fmin_l_bfgs_b as optim
from statsmodels.discrete.count_model import NegativeBinomialP, ZeroInflatedNegativeBinomialP, Poisson, ZeroInflatedPoisson
from scipy.stats import logistic, chi2

import tools.util as ut
import tools.scTransform as sct


# MB MLE estimation, from https://github.com/gokceneraslan/fit_nbinom/blob/master/fit_nbinom.py
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


def estimate_overdisp_nb(adata, layer=None, cutoff=0.01, flavor="sctransform"):

    if flavor == "BFGS":
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

    elif flavor == "sctransform":
        adata_sct = sct.SCTransform(adata,
                                    layer=layer,
                                    min_cells=1,
                                    gmean_eps=1,
                                    n_genes=2000,
                                    n_cells=None,  # use all cells
                                    bin_size=500,
                                    bw_adjust=3,
                                    inplace=False)
        adata.var["is_scd_outlier"] = adata_sct.var["is_scd_outlier"]
        adata.var["nb_overdisp"] = adata_sct.var["theta_sct"]
        adata.var["nb_overdisp_cutoff"] = adata.var["nb_overdisp"]
        adata.var["nb_overdisp_cutoff"][adata.var["nb_overdisp_cutoff"] < cutoff] = cutoff
        adata.var["nb_overdisp_cutoff"][np.isnan(adata.var["nb_overdisp_cutoff"])] = cutoff
        adata.var["nb_mean"] = adata_sct.var["Intercept_sct"]

    elif flavor == "moments":
        count_data = ut.convert_to_dense_counts(adata, layer)
        means = np.mean(count_data, axis=0)
        vars = np.var(count_data, axis=0)

        overdisps = means**2 / (vars - means)
        overdisps[overdisps <= 0] = 0.01

        adata.var["nb_mean"] = means
        adata.var["nb_overdisp"] = overdisps

    elif flavor == "statsmod_nb":
        count_data = ut.convert_to_dense_counts(adata, layer)
        n, p = count_data.shape

        overdisps = []
        means = []

        for i in range(p):

            if i % 100 == 0:
                print(f"gene {i}")

            dat = count_data[:, i].T

            model_nb = NegativeBinomialP(dat, np.ones(n))
            res_nb = model_nb.fit(method='bfgs', maxiter=5000, maxfun=5000, disp=0)
            means.append(np.exp(res_nb.params[0]))
            overdisps.append(1/res_nb.params[1])

        adata.var["nb_mean"] = means
        adata.var["nb_overdisp"] = overdisps

    elif flavor == "statsmod_auto":
        count_data = ut.convert_to_dense_counts(adata, layer)
        n, p = count_data.shape

        adata.var["gene_mean"] = np.mean(count_data, axis=0)
        adata.var["gene_var"] = np.var(count_data, axis=0)
        adata.var["mean_var_diff"] = adata.var["gene_mean"] - adata.var["gene_var"]
        adata.var["gene_dist"] = ["nb" if x < 0 else "poi" for x in adata.var["mean_var_diff"]]

        overdisps = []
        means = []
        zinf_params = []

        for i in range(p):

            if i % 100 == 0:
                print(f"gene {i}")

            dat = count_data[:, i].T
            dist = adata.var["gene_dist"][i]

            if dist == "poi":
                overdisps.append(np.inf)

                model_zipoi = ZeroInflatedPoisson(dat, np.ones(n))
                res_zipoi = model_zipoi.fit(method='bfgs', maxiter=5000, maxfun=5000, disp=0)

                model_poi = Poisson(dat, np.ones(n))
                res_poi = model_poi.fit(method='bfgs', maxiter=5000, maxfun=5000, disp=0)

                zipoi_loglik = res_zipoi.llf
                poi_loglik = res_poi.llf

                stat = 2 * (zipoi_loglik - poi_loglik)
                pvalue = 1 - chi2.ppf(stat, 1)

                if pvalue < 0.05:
                    means.append(np.exp(res_zipoi.params[1]))
                    zinf_params.append(logistic.pdf(res_zipoi.params[0]))
                else:
                    means.append(np.exp(res_poi.params[0]))
                    zinf_params.append(0.)

            elif dist == "nb":
                model_zinb = ZeroInflatedNegativeBinomialP(dat, np.ones(n))
                res_zinb = model_zinb.fit(method='bfgs', maxiter=5000, maxfun=5000, disp=0)

                model_nb = NegativeBinomialP(dat, np.ones(n))
                res_nb = model_nb.fit(method='bfgs', maxiter=5000, maxfun=5000, disp=0)

                zinb_loglik = res_zinb.llf
                nb_loglik = res_nb.llf

                stat = 2 * (zinb_loglik - nb_loglik)
                pvalue = 1 - chi2.ppf(stat, 1)

                if pvalue < 0.05 or np.isnan(nb_loglik):
                    means.append(np.exp(res_zinb.params[1]))
                    zinf_params.append(logistic.pdf(res_zinb.params[0]))
                    overdisps.append(1 / res_zinb.params[2])
                else:
                    means.append(np.exp(res_nb.params[0]))
                    zinf_params.append(0.)
                    overdisps.append(1 / res_nb.params[1])

        adata.var["est_mean"] = means
        adata.var["est_overdisp"] = overdisps
        adata.var["est_zero_inflation"] = zinf_params

