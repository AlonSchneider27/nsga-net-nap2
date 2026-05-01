"""Abstract base class for search spaces."""
from abc import ABC, abstractmethod

class SearchSpace(ABC):
    """Base class for NAS search spaces.
    Only needed during training phase. For inference, NAP2 accepts any nn.Module."""

    @abstractmethod
    def load_architectures(self, config=None):
        """Return list of (arch_id, arch_spec) pairs."""

    @abstractmethod
    def build_model(self, arch_spec):
        """Build a PyTorch nn.Module from an architecture spec."""

    def get_model_info(self, arch_spec):
        """Optional: human-readable description."""
        return str(arch_spec)
