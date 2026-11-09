"""Compatibility wrapper for pipeline.fault."""
from pipeline.fault import (
    calibrate,
    calibrate_v4,
    new_events,
    unavailability,
    B_SPAN,
    V4_SVC_SIGNALS,
    SVC_SIGNALS,
)

__all__ = [
    "calibrate",
    "calibrate_v4",
    "new_events",
    "unavailability",
    "B_SPAN",
    "V4_SVC_SIGNALS",
    "SVC_SIGNALS",
]
