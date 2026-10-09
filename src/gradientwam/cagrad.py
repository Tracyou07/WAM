"""Independent, stdlib-only implementation of two-task unrescaled CAGrad.

Reference: Liu et al., NeurIPS 2021, Eq. (3) / Algorithm 1,
https://arxiv.org/abs/2110.14048 . No upstream implementation is copied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cagrad_coefficients(
    gram: Sequence[Sequence[float]], c: float = 0.4
) -> tuple[float, float]:
    """Return coefficients for ``d = coeff[0] * gv + coeff[1] * ga``.

    ``gram[i][j]`` is the dot product of the two already weighted,
    accumulated and globally averaged task gradients on common generator
    parameters only. Coefficients are ordinary Python floats, not simplex
    probabilities: they need not sum to one. No normalization or rescaling
    of the returned direction is performed.

    The input must be a finite symmetric positive semidefinite 2x2 matrix
    and ``0 <= c < 1``. Relative roundoff up to 1e-12 is symmetrized/clamped;
    larger violations raise ValueError. Compute dot products in float64.
    Zero mean or either zero task gradient returns the mean coefficients;
    in the latter case the primal optimum is nonunique and the mean is an
    optimal feasible choice. See docs/cagrad_reference.md for boundaries.
    """
    try:
        c = float(c)
        if len(gram) != 2 or any(len(row) != 2 for row in gram):
            raise ValueError("gram must have shape (2, 2)")
        values = [[float(value) for value in row] for row in gram]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gram must be a real 2x2 matrix and c a real scalar") from exc
    if not math.isfinite(c) or not 0.0 <= c < 1.0:
        raise ValueError("c must be finite and in [0, 1)")
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("gram must contain finite values")

    # Common scaling leaves the coefficients unchanged and prevents overflow.
    scale = max(abs(value) for row in values for value in row)
    if scale == 0.0:
        return (0.5, 0.5)
    a, upper = (value / scale for value in values[0])
    lower, b = (value / scale for value in values[1])
    tolerance = 1e-12
    if abs(upper - lower) > tolerance:
        raise ValueError("gram must be symmetric")
    h = (upper + lower) / 2.0
    if a < -tolerance or b < -tolerance:
        raise ValueError("gram must be positive semidefinite")
    a, b = max(a, 0.0), max(b, 0.0)
    bound = math.sqrt(a) * math.sqrt(b)
    if abs(h) > bound + tolerance:
        raise ValueError("gram must be positive semidefinite")
    h = max(-bound, min(bound, h))
    mean_sq = max(0.0, math.fsum((a, b, 2.0 * h)) / 4.0)
    if c == 0.0 or mean_sq == 0.0 or a == 0.0 or b == 0.0:
        return (0.5, 0.5)

    # gw = x*gv + (1-x)*ga. The dual is linear + radius*||gw||.
    span_sq = max(0.0, math.fsum((a, b, -2.0 * h)))
    if span_sq == 0.0:
        return ((1.0 + c) / 2.0, (1.0 + c) / 2.0)
    radius = c * math.sqrt(mean_sq)
    slope = (a - b) / 2.0
    limit = radius * math.sqrt(span_sq)
    if slope >= limit:
        x, weighted_sq = 0.0, b
    elif slope <= -limit:
        x, weighted_sq = 1.0, a
    else:
        center = (b - h) / span_sq
        minimum_sq = max(0.0, math.fsum((a * b, -h * h)) / span_sq)
        ratio = -slope / limit
        offset = ratio * math.sqrt(minimum_sq / span_sq / (1.0 - ratio * ratio))
        x = min(1.0, max(0.0, center + offset))
        weighted_sq = minimum_sq + span_sq * (x - center) ** 2
    if weighted_sq == 0.0:
        return (0.5, 0.5)
    multiplier = radius / math.sqrt(weighted_sq)
    return (0.5 + multiplier * x, 0.5 + multiplier * (1.0 - x))
