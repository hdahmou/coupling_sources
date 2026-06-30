import jax
import numpyro

from coupling_sources.model import choice_model


def fit_mcmc(
    key: jax.Array,
    choices: jax.Array,
    source_choices: jax.Array,
    hidden_source: jax.Array,
    num_warmup: int = 300,
    num_samples: int = 600,
    num_chains: int = 1,
    chain_method: str = "parallel",
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 10,
    progress_bar: bool = False,
) -> numpyro.infer.MCMC:
    data = {
        "choices": choices,
        "source_choices": source_choices,
        "hidden_source": hidden_source,
    }
    kernel = numpyro.infer.NUTS(
        choice_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(key, data, None, extra_fields=("diverging",))
    return mcmc
