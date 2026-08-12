"""Canonical parameter codec for the inner-tuner benchmark (PLAN §3.1).

All new arms share one encoding into normalized space:

- numeric dimensions (float + int, declaration order) map to z ∈ [0,1];
- log-scale floats are linearly normalized in log space
  (z = (log v − log lo)/(log hi − log lo));
- int inverse transform: deterministic nearest rounding (half-up), then clamp
  to [lo, hi];
- float inverse transform: z is clamped into [0,1] first (projection Π), and
  the decoded value is clamped into [lo, hi] (log space overshoots the bound
  by 1-2 ULP at the endpoints, which Π makes attractors);
- categorical dimensions are NOT part of the z vector — they keep semantic
  labels everywhere, with encode_categorical for numeric models.

Postcondition: decode() output is always inside the declared search space —
_deterministic_preflight must never be able to return out_of_space for a
config this codec produced.

Deterministic; no RNG inside. Decoded params come out in canonical
(declaration) key order, cast through the production cast.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

import tune_tools  # noqa: E402
from _common import cast_params_to_search_space  # noqa: E402


class Codec:
    """Bidirectional map between valid params dicts and normalized z space.

    Built from a space.CandidateContract. Per-dimension metadata is exposed
    (numeric_dimensions / continuous_dimensions / categorical_dimensions) so
    arms can select subsets of the z vector (e.g. continuous-only for SPSA)
    by index over the declared numeric order.
    """

    def __init__(self, contract) -> None:
        self.contract = contract
        self.numeric_dimensions = contract.numeric_dimensions
        self.continuous_dimensions = contract.continuous_dimensions
        self.categorical_dimensions = contract.categorical_dimensions
        self.z_dim = len(self.numeric_dimensions)

    def project(self, z) -> np.ndarray:
        """Projection Π: clip into [0,1]^d."""
        return np.clip(np.asarray(z, dtype=float), 0.0, 1.0)

    def encode(self, params: dict) -> tuple[np.ndarray, dict]:
        """params -> (z, cat_labels) over numeric dims in declaration order.

        params must be valid (in-bounds) contract values; out-of-bounds values
        map outside [0,1] honestly (no silent projection on encode) —
        including on degenerate dimensions, where any value other than the
        single legal one maps outside the range. A non-positive value on a
        log-scale dimension raises ValueError naming the dimension.
        """
        z = np.array(
            [self._encode_dimension(dim, params[dim.name])
             for dim in self.numeric_dimensions],
            dtype=float,
        )
        cat_labels = {dim.name: params[dim.name]
                      for dim in self.categorical_dimensions}
        return z, cat_labels

    def decode(self, z, cat_labels: dict) -> dict:
        """(z, cat_labels) -> full valid params dict.

        z is projected into [0,1] first; categorical labels must cover every
        categorical dimension with an in-options label. The result is cast
        through the production cast and ordered canonically.
        """
        z = self.project(z)
        if z.shape[0] != self.z_dim:
            raise ValueError(f"z has length {z.shape[0]}, expected {self.z_dim}")
        missing = [dim.name for dim in self.categorical_dimensions
                   if dim.name not in cat_labels]
        if missing:
            raise ValueError(f"missing categorical labels: {missing}")
        params: dict = {}
        index = 0
        for dim in self.contract.dimensions:
            if dim.kind == "categorical":
                label = cat_labels[dim.name]
                if not tune_tools._categorical_contains(dim.options, label):
                    raise ValueError(
                        f"label {label!r} not in options of dimension {dim.name!r}"
                    )
                params[dim.name] = label
            else:
                params[dim.name] = self._decode_dimension(dim, float(z[index]))
                index += 1
        return cast_params_to_search_space(params, self.contract.search_space)

    def encode_categorical(self, name: str, label) -> int:
        """Semantic label -> option index of one categorical dimension."""
        for dim in self.categorical_dimensions:
            if dim.name == name:
                for index, option in enumerate(dim.options):
                    if tune_tools._categorical_value_equal(option, label):
                        return index
                raise ValueError(
                    f"label {label!r} not in options of dimension {name!r}"
                )
        raise ValueError(f"no categorical dimension named {name!r}")

    @staticmethod
    def _encode_dimension(dim, value) -> float:
        value = float(value)
        if dim.hi == dim.lo:
            # Degenerate: [0,1] carries no information here. 0.0 for lo itself,
            # and out-of-range values still land outside [0,1] so the docstring
            # promise holds for them too.
            if value == dim.lo:
                return 0.0
            return -1.0 if value < dim.lo else 2.0
        if dim.kind == "float" and dim.log:
            if value <= 0.0:
                raise ValueError(
                    f"dimension {dim.name!r} is log-scale; cannot encode "
                    f"non-positive value {value!r}"
                )
            return (math.log(value) - math.log(dim.lo)) / (
                math.log(dim.hi) - math.log(dim.lo)
            )
        return (value - dim.lo) / (dim.hi - dim.lo)

    @staticmethod
    def _decode_dimension(dim, z: float):
        if dim.kind == "int":
            value = dim.lo + z * (dim.hi - dim.lo)
            nearest = int(math.floor(value + 0.5))  # deterministic nearest, half-up
            return min(max(nearest, dim.lo), dim.hi)
        if dim.hi == dim.lo:
            return float(dim.lo)
        if dim.log:
            value = math.exp(
                math.log(dim.lo) + z * (math.log(dim.hi) - math.log(dim.lo))
            )
        else:
            value = dim.lo + z * (dim.hi - dim.lo)
        # exp/log (and, rarely, the linear form) overshoot the declared bound by
        # 1-2 ULP at z in {0,1} — and Pi makes those endpoints attractors, not
        # rare events. Clamp in value space, mirroring the int branch.
        return min(max(value, dim.lo), dim.hi)
