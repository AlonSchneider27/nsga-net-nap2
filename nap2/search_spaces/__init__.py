"""Search space registry."""
from nap2.search_spaces.base import SearchSpace

SEARCH_SPACES = {}

def register_search_space(name):
    def decorator(cls):
        SEARCH_SPACES[name] = cls
        return cls
    return decorator

def create_search_space(name, **kwargs):
    if name not in SEARCH_SPACES:
        raise ValueError(f"Unknown search space: {name}. Available: {list(SEARCH_SPACES.keys())}")
    return SEARCH_SPACES[name](**kwargs)

# Import implementations to trigger registration
try:
    from nap2.search_spaces.nasbench201 import NASBench201SearchSpace
except ImportError:
    pass  # xautodl not installed
