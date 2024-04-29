import logging
import timeit

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import jax
import gpjax as gpx
import optax as ox

import jax.numpy as jnp
import jax.random as jr

log = logging.getLogger(__name__)

from gpjax.typing import ScalarFloat
from cola.linalg.decompositions.decompositions import Cholesky
import cola
from fasthgp.hgp import HGP
from fasthgp.utils import save_model

class HGPobjective(gpx.objectives.AbstractObjective):
    def step(self, model: HGP, train_data: gpx.Dataset) -> ScalarFloat:
        B = model.B
        s2 = model.likelihood.obs_stddev**2
        lambda_j = model.bf.eigenvalues()
        S = jax.vmap(model.prior.kernel.spectral_density, 0, 0)(jnp.sqrt(lambda_j))
        Lambdainv = jnp.diag(1/S)
        Z = s2 * Lambdainv + B
        Z += cola.ops.I_like(Z) * model.prior.jitter
        Z = cola.PSD(Z)
        logdetZ = cola.linalg.logdet(Z, Cholesky())
        logdetQ = (train_data.n - model.M) * jnp.log(s2) + logdetZ + jnp.log(S).sum()
        aZa = model.alpha.T @ cola.solve(Z, model.alpha, Cholesky())
        yTy = train_data.y.T @ train_data.y
        yTQy = 1/s2 * (yTy - aZa)
        const = train_data.n * jnp.log(2 * jnp.pi)
        return 1/2 * (logdetQ + yTQy + const).squeeze()
objective = jax.jit(HGPobjective(negative=True))
key = jr.PRNGKey(13)

setup = \
"""import jax
import gpjax as gpx
def train_loop(model, inp):
    xi, yi = inp
    Di = gpx.Dataset(xi[None, :], yi[None, :])
    model = model.update_with_batch(Di)
    return model, None
"""

def eval_data(alg, train_data):
    log.parent.disabled = True
    # Update w/ data
    alg = alg.update_with_batch(train_data)
    func = \
"""
jax.lax.scan(train_loop, alg, [train_data.X, train_data.y])
alg.B
"""

    t = timeit.repeat(func,
                      repeat=15,
                      number=1,
                      globals=locals(),
                      setup=setup)
    result = dict(
        times = t,
        m = alg.M,
    )
    alg, _ = gpx.fit(model=alg,
                    objective=objective,
                    train_data=train_data,
                    optim=ox.adam(learning_rate=1e-1),
                    num_iters=100,
                    key=key)
    return alg, result

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", True)
    log.info("Instantiating objects")
    data_generator = instantiate(cfg.data_generator)
    alg = instantiate(cfg.alg)
    log.info("Generating data")
    D = data_generator()
    log.info("Data generated!")
    alg, result = eval_data(alg, D)
    log.info("Evaluation complete")
    log.info("Saving data and quitting")
    jnp.savez('result.npz',
             **result)
    save_model(alg, "model")
    
if __name__ == "__main__":
    main()
