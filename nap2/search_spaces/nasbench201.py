"""NAS-Bench-201 search space implementation."""
from nap2.search_spaces import register_search_space
from nap2.search_spaces.base import SearchSpace

@register_search_space("nasbench201")
class NASBench201SearchSpace(SearchSpace):
    """NAS-Bench-201 search space. Requires xautodl package."""

    # Dataset name -> number of classes
    DATASET_NUM_CLASSES = {
        "cifar10": 10,
        "cifar100": 100,
        "ImageNet16-120": 120,
    }

    def __init__(self, api_path=None, dataset="cifar10"):
        self.api_path = api_path
        self.dataset = dataset
        self.num_classes = self.DATASET_NUM_CLASSES.get(dataset, 10)
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from nats_bench import create as nats_create
                self._api = nats_create(self.api_path, search_space="topology", fast_mode=True)
            except ImportError:
                raise ImportError(
                    "NAS-Bench-201 requires xautodl and nats_bench. "
                    "Install with: pip install xautodl nats_bench"
                )
        return self._api

    def load_architectures(self, config=None):
        api = self._get_api()
        max_models = config.get("max_models") if config else None
        total = len(api) if max_models is None else min(max_models, len(api))
        return [(idx, api[idx]) for idx in range(total)]

    def build_model(self, arch_spec, num_classes=None):
        """Build a NAS-Bench-201 model.

        Args:
            arch_spec: architecture string from NAS-Bench-201 API
            num_classes: override number of output classes
                (default: determined by dataset passed to __init__)
        """
        from xautodl.models import get_cell_based_tiny_net
        from xautodl.config_utils import dict2config
        nc = num_classes or self.num_classes
        config = dict2config({"name": "infer.tiny", "C": 16, "N": 5,
                              "arch_str": arch_spec, "num_classes": nc}, None)
        return get_cell_based_tiny_net(config)
