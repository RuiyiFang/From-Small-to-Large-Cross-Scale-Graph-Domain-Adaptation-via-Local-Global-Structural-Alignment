"""LGSA: Local-Global Structural Alignment for Cross-Scale Graph Domain Adaptation."""

from lgsa.cross_scale import (
    CrossScaleConfig,
    sample_cross_scale_source,
    cross_scale_statistics,
)

__version__ = "1.0.0"

__all__ = [
    "CrossScaleConfig",
    "sample_cross_scale_source",
    "cross_scale_statistics",
]
