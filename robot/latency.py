"""Latency percentile summaries in the shape every benchmark report uses."""


def percentiles(values, quantiles=(50, 95, 99)):
    """Return {'p50': ..., 'p95': ..., 'p99': ...}, or None for no samples."""
    import numpy as np
    if len(values) == 0:
        return None
    return {f'p{q}': float(np.percentile(values, q)) for q in quantiles}
