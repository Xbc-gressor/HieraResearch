"""Candidate contract reading for the inner-tuner benchmark.

Reads PARAM_SCHEMA / SEARCH_SPACE / BASE_PARAMS from a candidate's train.py
by AST literal_eval — the candidate is never imported. Readers and validators
are reused from production (tools/tuners/tune_tools.py, tools/tuners/_common.py).
SEARCH_SPACE declaration order is the canonical dimension order for the whole
benchmark; key order is semantically meaningful (part of execution revision).

Scores are always lower-is-better.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

import tune_tools  # noqa: E402
from _common import (  # noqa: E402
    cast_params_to_search_space,
    params_identity as _json_identity,
)

IDENTITY_SIGNIFICANT_DIGITS = 12


def _quantize_floats(params: dict) -> dict:
    """Round float values to IDENTITY_SIGNIFICANT_DIGITS significant digits."""
    out = {}
    for key, value in params.items():
        if isinstance(value, float) and value != 0.0 and math.isfinite(value):
            exponent = int(math.floor(math.log10(abs(value))))
            value = round(value, -exponent + IDENTITY_SIGNIFICANT_DIGITS - 1)
        out[key] = value
    return out


@dataclass(frozen=True)
class Dimension:
    """One search-space dimension, in canonical (declaration) order."""

    name: str
    kind: str  # "float" | "int" | "categorical"
    log: bool = False
    lo: float | int | None = None
    hi: float | int | None = None
    options: tuple | None = None

    @property
    def is_degenerate(self) -> bool:
        """True when this dimension admits exactly one value.

        Degenerate dimensions silently absorb perturbations: decode returns the
        same value for every z, so an arm that moves one produces a no-op (and,
        for SPSA, a spurious nonzero gradient component along a direction that
        cannot move). Arms should exclude them — see varying_dimensions.
        """
        if self.kind == "categorical":
            return len(self.options) == 1
        return self.lo == self.hi


@dataclass(frozen=True)
class CandidateContract:
    """Parsed and validated tuner contract of one frozen candidate."""

    path: Path
    param_schema: dict
    search_space: dict  # declaration order = canonical dimension order
    base_params: dict
    dimensions: tuple[Dimension, ...]

    @property
    def numeric_dimensions(self) -> tuple[Dimension, ...]:
        """Float + int dimensions, declaration order."""
        return tuple(d for d in self.dimensions if d.kind in ("float", "int"))

    @property
    def continuous_dimensions(self) -> tuple[Dimension, ...]:
        """Float dimensions only, declaration order."""
        return tuple(d for d in self.dimensions if d.kind == "float")

    @property
    def categorical_dimensions(self) -> tuple[Dimension, ...]:
        return tuple(d for d in self.dimensions if d.kind == "categorical")

    @property
    def varying_dimensions(self) -> tuple[Dimension, ...]:
        """Dimensions that admit more than one value, declaration order.

        The subset arms should actually move; see Dimension.is_degenerate.
        """
        return tuple(d for d in self.dimensions if not d.is_degenerate)

    def cast(self, params: dict) -> dict:
        """Production canonical cast (_common.cast_params_to_search_space)."""
        return cast_params_to_search_space(dict(params), self.search_space)

    def params_identity(self, params: dict) -> str:
        """Canonical identity: production cast + float quantization + production
        stable JSON identity.

        Same semantics as production exact-duplicate detection
        (tune_tools.validate_proposals, _common.attempted_config_identities):
        cast first, then identity. Key order cannot evade detection (the
        identity is sorted); cast-equivalent values (e.g. 3 vs 3.0 on an int
        dimension) are identical — note production int cast truncates toward
        zero, so 3.7 on an int dimension is identical to 3.

        SANCTIONED DEVIATION from production: floats are quantized to
        IDENTITY_SIGNIFICANT_DIGITS before hashing. Production compares raw
        JSON, but every z-space arm reaches its configs through
        codec.decode(codec.encode(...)), whose log-space round trip perturbs
        values by ~1 ULP. Under exact identity that made an arm's own anchor a
        non-duplicate (measured: 88.9% of real anchors), so it would re-consume
        budget re-measuring a known point and could take the incumbent on
        measurement noise alone. Quantization makes "same config" mean the same
        thing to the codec and to duplicate detection. Measured on the full
        real corpus (333 contract/anchor pairs): round-trip mismatch 0, and no
        two genuinely distinct configs collapse.
        """
        return _json_identity(_quantize_floats(self.cast(params)))

    def is_duplicate(self, params: dict, history) -> bool:
        """True when params' canonical identity appears in ``history``
        (an iterable of params dicts). Duplicate = identical identity after cast."""
        identity = self.params_identity(params)
        return any(self.params_identity(seen) == identity for seen in history)


def read_contract(candidate_path) -> CandidateContract:
    """Parse and validate the candidate contract from its train.py path.

    Raises ValueError on a malformed contract: missing/duplicate/non-literal
    mappings (via the production AST reader), or any violated relational
    invariant from production lint_contract (key-set equality across
    PARAM_SCHEMA / SEARCH_SPACE / BASE_PARAMS, valid entries, schema/space
    agreement, in-bounds BASE_PARAMS, make_model def). OSError/SyntaxError
    propagate for unreadable or unparseable files.
    """
    path = Path(candidate_path)
    param_schema = tune_tools._read_literal_mapping(path, "PARAM_SCHEMA")
    search_space = tune_tools._read_literal_mapping(path, "SEARCH_SPACE")
    base_params = tune_tools._read_literal_mapping(path, "BASE_PARAMS")
    lint = tune_tools.lint_contract(path)
    if not lint["ok"]:
        details = "; ".join(
            f"{error['code']}: {error['detail']}" for error in lint["errors"]
        )
        raise ValueError(f"invalid candidate contract in {path}: {details}")
    dimensions = tuple(
        _dimension(name, entry) for name, entry in search_space.items()
    )
    return CandidateContract(
        path=path,
        param_schema=param_schema,
        search_space=search_space,
        base_params=base_params,
        dimensions=dimensions,
    )


def with_search_space(
    contract: CandidateContract, search_space: dict
) -> CandidateContract:
    """Rebind a contract to a (clamped) search space, rebuilding dimensions.

    Production engines shrink the box with clamp_search_space_to_preflight
    before searching; the arm-visible contract must show the same bounds —
    bounds checks, the codec, and the proposer's rendered search space all
    read them from the contract.
    """
    return CandidateContract(
        path=contract.path,
        param_schema=contract.param_schema,
        search_space=search_space,
        base_params=contract.base_params,
        dimensions=tuple(
            _dimension(name, entry) for name, entry in search_space.items()
        ),
    )


def _dimension(name: str, entry) -> Dimension:
    """Build a Dimension from a lint-validated SEARCH_SPACE entry."""
    kind = entry[0]
    if kind == "float":
        return Dimension(
            name=name,
            kind="float",
            log=len(entry) == 4,
            lo=float(entry[1]),
            hi=float(entry[2]),
        )
    if kind == "int":
        return Dimension(name=name, kind="int", lo=int(entry[1]), hi=int(entry[2]))
    return Dimension(name=name, kind="categorical", options=tuple(entry[1]))
