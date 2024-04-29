import jax
jax.config.update("jax_enable_x64", True)
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
from fasthgp.shgp import SHGP
from fasthgp.hgp import HGP
from fasthgp.kernels import SE, LaplaceBF
from fasthgp.utils import gamma
key = jr.PRNGKey(13)
import fasthgp.examples.random_example as re
from tqdm import tqdm
from timeit import repeat

N = 1
Ls = 2.0*jnp.ones(3,)
ms = jnp.array([15, 15, 15])
bf = LaplaceBF(num_bfs=ms, L=Ls)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=N, obs_stddev=jnp.array([0.01]))
meanf = gpx.mean_functions.Zero()
kernel = SE(lengthscale=.2, variance=0.25)
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
D, Dtest = re.generate_data(gpx.kernels.RBF(lengthscale=kernel.lengthscale, variance=kernel.variance), N=N, key=key)



mi = jnp.arange(2, 40)
models = {'SHGP': SHGP, 'HGP': HGP}
results = {model: dict(m=[], t=[]) for model in models.keys()}
for m in tqdm(mi, desc="m="):
    ms = jnp.ones((3,), dtype=int) * m
    bf = LaplaceBF(num_bfs=ms, L=Ls)
    shgp = SHGP(prior, likelihood, bf=bf)
    gamma(D.X, shgp.unique_k, shgp.bf.L)
    tshgp = repeat('gamma(D.X, shgp.unique_k, shgp.bf.L)',
                   repeat=10,
                   number=500, # "Number of measurements"
                   globals=locals())
    

    if m < 25:
        # After 24, the standard HGP runs into memory issues
        hgp = HGP(prior, likelihood, bf=bf)
        bf(D.X)
        hgpfun=\
"""
phi = bf(D.X)
phi.T @ phi
"""
        thgp = repeat(hgpfun,
                  repeat=10,
                  number=500, # "Number of measurements"
                  globals=locals())
        
        results['HGP']['t'].append(thgp[1:]) # The first time is often an outlier
        results['HGP']['m'].append(bf.M)

    results['SHGP']['t'].append(tshgp[1:]) # The first time is often an outlier
    results['SHGP']['m'].append(bf.M)
    

# ### Save timing data
# ##### Save raw data as json

for model in results.keys():
    results[model]['m'] = [int(x) for x in results[model]['m']]

import json
with open("hankel_timings.json", "w") as file:
    json.dump(results, file, indent=4)


# ##### Save curated data to .csv for tikz plotting

to_csv = {model: dict() for model in results.keys()}
for model, mres in results.items():
    t = jnp.array(mres['t'])
    to_csv[model]['t_median'] = jnp.median(t, axis=1)
    to_csv[model]['t_min'] = jnp.min(t, axis=1)
    to_csv[model]['t_max'] = jnp.max(t, axis=1)
    to_csv[model]['t_mean'] = jnp.mean(t, axis=1)
    to_csv[model]['t_std'] = jnp.std(t, axis=1)
    to_csv[model]['m'] = jnp.array(mres['m'])


import pandas as pd
for model, val in to_csv.items():
    D = pd.DataFrame(val)
    with open(model+'timing.csv', "w") as file:
        D.to_csv(file, index=False)
