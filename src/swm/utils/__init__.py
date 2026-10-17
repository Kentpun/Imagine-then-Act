from .config import load_config

__all__ = ["load_config", "set_seed"]


def set_seed(seed: int) -> None:
    from .seed import set_seed as _set_seed

    _set_seed(seed)
