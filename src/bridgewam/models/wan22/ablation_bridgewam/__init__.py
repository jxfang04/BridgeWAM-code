"""Isolated BridgeWAM ablation models.

The directory name intentionally follows the experiment namespace requested for
these one-off causal ablations. Production BridgeWAM code remains in the parent
package.
"""

from .factory import create_bridgewam_ablation

__all__ = ["create_bridgewam_ablation"]
