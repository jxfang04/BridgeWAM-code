"""BridgeWAM training and simulation evaluation."""
__all__ = ["BridgeWAM"]


def __getattr__(name):
    if name == "BridgeWAM":
        from .models.wan22.bridgewam import BridgeWAM
        return BridgeWAM
    raise AttributeError(name)
