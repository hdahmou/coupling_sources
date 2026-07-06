from __future__ import annotations

from contextlib import contextmanager

import numpyro.infer.mcmc as numpyro_mcmc
import numpyro.util as numpyro_util


class _LabeledTqdm:
    def __init__(self, bar: object, label: str):
        self.bar = bar
        self.label = label

    def set_description(self, description: object, *args: object, **kwargs: object) -> object:
        text = str(description).strip()
        labeled = f"[{self.label}] {text}" if text else f"[{self.label}]"
        return self.bar.set_description(labeled, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.bar, name)


@contextmanager
def labeled_progress_bar(label: str | None):
    if label is None:
        yield
        return
    original_fori_collect = numpyro_mcmc.fori_collect
    original_tqdm_auto = numpyro_util.tqdm_auto

    def with_label(description: object) -> str:
        text = str(description).strip()
        return f"[{label}] {text}" if text else f"[{label}]"

    def fori_collect_with_label(*args: object, **kwargs: object) -> object:
        original_desc = kwargs.get("progbar_desc", lambda _: "")
        kwargs["progbar_desc"] = lambda index: with_label(original_desc(index))
        return original_fori_collect(*args, **kwargs)

    def tqdm_auto_with_label(*args: object, **kwargs: object) -> _LabeledTqdm:
        return _LabeledTqdm(original_tqdm_auto(*args, **kwargs), label)

    numpyro_mcmc.fori_collect = fori_collect_with_label
    numpyro_util.tqdm_auto = tqdm_auto_with_label
    try:
        yield
    finally:
        numpyro_mcmc.fori_collect = original_fori_collect
        numpyro_util.tqdm_auto = original_tqdm_auto
