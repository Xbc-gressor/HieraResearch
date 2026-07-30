from __future__ import annotations

import itertools
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import apply_base_params  # noqa: E402
import apply_search_space  # noqa: E402
from _common import load_candidate_modules  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    _expand_space_entry,
    _read_search_space,
    _schema_accepts_value,
    _valid_schema_entry,
    check_search_space,
    lint_contract,
    lint_schema,
    valid_space_entry,
)


SCHEMA = {
    "depth": "int",
    "rate": ("float", "log"),
    "mode": ("categorical", ["allowed", "other"]),
}


class ContractIdentityTests(unittest.TestCase):
    def _candidate(self, source: str) -> tuple[tempfile.TemporaryDirectory, Path]:
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "train.py"
        path.write_text(source)
        return tmp, path

    def test_duplicate_runtime_binding_is_rejected_everywhere(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3)}
BASE_PARAMS = {"x": 1}
def make_model(params):
    return params
SEARCH_SPACE = {"y": ("int", 1, 3)}
BASE_PARAMS = {"y": 2}
def make_model(params):
    return {"runtime": params}
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_contract(path)
        duplicate_names = {
            error["detail"].split()[0]
            for error in verdict["errors"]
            if error["code"] == "duplicate_symbol"
        }
        self.assertEqual(
            duplicate_names,
            {"SEARCH_SPACE", "BASE_PARAMS", "make_model"},
        )
        with self.assertRaisesRegex(SystemExit, "exactly one"):
            _read_search_space(path)
        with self.assertRaisesRegex(SystemExit, "exactly one"):
            apply_base_params.apply(path, {"x": 3})
        with self.assertRaisesRegex(SystemExit, "exactly one"):
            apply_search_space.apply(path, {"x": ["int", 1, 4]})

    def test_schema_mode_rejects_materialized_contract_symbols(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3)}
BASE_PARAMS = {"x": 1}
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_schema(path)
        self.assertFalse(verdict["ok"])
        self.assertEqual(
            {
                error["detail"].split()[0]
                for error in verdict["errors"]
                if error["code"] == "stray_symbol"
            },
            {"SEARCH_SPACE", "BASE_PARAMS"},
        )

    def test_contract_rejects_schema_mode_and_invalid_log_bounds(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": ("float", "log")}
SEARCH_SPACE = {"x": ("float", 0.0, 1.0, "log")}
BASE_PARAMS = {"x": 0.5}
async def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_contract(path)
        self.assertFalse(verdict["ok"])
        self.assertIn("bad_tuple", {error["code"] for error in verdict["errors"]})
        self.assertIn(
            "make_model_not_func",
            {error["code"] for error in verdict["errors"]},
        )

    def test_module_scope_contract_mutation_is_rejected(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3)}
SEARCH_SPACE.update({"y": ("int", 1, 2)})
BASE_PARAMS = {"x": 1}
if True:
    BASE_PARAMS = {"x": 2}
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_contract(path)

        self.assertFalse(verdict["ok"])
        self.assertGreaterEqual(
            sum(
                error["code"] == "contract_mutation"
                for error in verdict["errors"]
            ),
            2,
        )

    def test_duplicate_dict_key_is_rejected_instead_of_last_wins(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {
    "x": ("int", 1, 3),
    "x": ("int", 4, 6),
}
BASE_PARAMS = {"x": 5}
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_contract(path)

        self.assertFalse(verdict["ok"])
        self.assertTrue(
            any("duplicate key" in error["detail"] for error in verdict["errors"])
        )
        with self.assertRaisesRegex(SystemExit, "duplicate key"):
            _read_search_space(path)

    def test_chained_contract_declarations_are_rejected_without_rewrite(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = SPACE_ALIAS = {"x": ("int", 1, 3)}
BASE_PARAMS = DEFAULT_PARAMS = {"x": 1}
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)
        before = path.read_bytes()

        verdict = lint_contract(path)

        self.assertFalse(verdict["ok"])
        self.assertGreaterEqual(
            sum(
                error["code"] == "non_simple_declaration"
                for error in verdict["errors"]
            ),
            2,
        )
        with self.assertRaisesRegex(SystemExit, "simple module-level"):
            _read_search_space(path)
        with self.assertRaisesRegex(SystemExit, "simple module-level"):
            apply_base_params.apply(path, {"x": 2})
        with self.assertRaisesRegex(SystemExit, "simple module-level"):
            apply_search_space.apply(path, {"x": ["int", 1, 4]})
        self.assertEqual(path.read_bytes(), before)

    def test_module_scope_mapping_alias_escape_is_rejected(self) -> None:
        tmp, path = self._candidate(
            """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3)}
BASE_PARAMS = {"x": 1}
space_alias = SEARCH_SPACE
space_alias["y"] = ("int", 1, 2)
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)

        verdict = lint_contract(path)

        self.assertFalse(verdict["ok"])
        self.assertIn(
            "contract_alias_escape",
            {error["code"] for error in verdict["errors"]},
        )

    def test_malformed_schema_returns_diagnostics_in_both_modes(self) -> None:
        tmp, complete = self._candidate(
            """
PARAM_SCHEMA = {"x": []}
SEARCH_SPACE = {"x": ("int", 1, 3)}
BASE_PARAMS = {"x": 1}
def make_model(params):
    return params
""".lstrip()
        )
        self.addCleanup(tmp.cleanup)
        schema_only = complete.parent / "schema_only.py"
        schema_only.write_text(
            """
PARAM_SCHEMA = {"x": []}
def make_model(params):
    return params
""".lstrip()
        )

        complete_verdict = lint_contract(complete)
        schema_verdict = lint_schema(schema_only)

        self.assertFalse(complete_verdict["ok"])
        self.assertFalse(schema_verdict["ok"])
        self.assertIn(
            "bad_schema_entry",
            {error["code"] for error in complete_verdict["errors"]},
        )
        self.assertEqual(schema_verdict["kinds"], {})


class RuntimeContractIdentityTests(unittest.TestCase):
    SOURCE = """
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, %d)}
BASE_PARAMS = {"x": 1}
def make_model(params):
    return params
""".lstrip()

    def _candidate(self, source: str) -> tuple[tempfile.TemporaryDirectory, Path]:
        tmp = tempfile.TemporaryDirectory()
        directory = Path(tmp.name)
        (directory / "prepare.py").write_text("")
        path = directory / "train.py"
        path.write_text(source)
        return tmp, path

    def test_loader_ignores_same_timestamp_same_size_stale_bytecode(self) -> None:
        tmp, path = self._candidate(self.SOURCE % 3)
        self.addCleanup(tmp.cleanup)
        py_compile.compile(str(path), doraise=True)
        before = path.stat()

        path.write_text(self.SOURCE % 4)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        train_module, _ = load_candidate_modules(path)

        self.assertEqual(train_module.SEARCH_SPACE["x"], ("int", 1, 4))

    def test_runtime_contract_mutation_cannot_diverge_from_ast(self) -> None:
        source = self.SOURCE % 3
        source += (
            'namespace = vars(__import__(__name__))\n'
            'namespace["SEARCH_SPACE"]["x"] = ("int", 1, 4)\n'
        )
        tmp, path = self._candidate(source)
        self.addCleanup(tmp.cleanup)
        self.assertTrue(lint_contract(path)["ok"])

        with self.assertRaisesRegex(
            RuntimeError,
            "runtime SEARCH_SPACE differs from its AST literal",
        ):
            load_candidate_modules(path)

    def test_runtime_contract_reordering_is_not_treated_as_identical(self) -> None:
        source = """
PARAM_SCHEMA = {"x": "int", "y": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3), "y": ("int", 1, 3)}
BASE_PARAMS = {"x": 1, "y": 1}
def make_model(params):
    return params
namespace = vars(__import__(__name__))
namespace["SEARCH_SPACE"] = dict(
    reversed(namespace["SEARCH_SPACE"].items())
)
""".lstrip()
        tmp, path = self._candidate(source)
        self.addCleanup(tmp.cleanup)
        self.assertTrue(lint_contract(path)["ok"])

        with self.assertRaisesRegex(
            RuntimeError,
            "runtime SEARCH_SPACE differs from its AST literal",
        ):
            load_candidate_modules(path)

    def test_search_space_order_is_part_of_execution_revision(self) -> None:
        first = """
PARAM_SCHEMA = {"x": "int", "y": "int"}
SEARCH_SPACE = {"x": ("int", 1, 3), "y": ("int", 1, 3)}
BASE_PARAMS = {"x": 1, "y": 1}
def make_model(params):
    return params
""".lstrip()
        second = first.replace(
            'SEARCH_SPACE = {"x": ("int", 1, 3), "y": ("int", 1, 3)}',
            'SEARCH_SPACE = {"y": ("int", 1, 3), "x": ("int", 1, 3)}',
        )
        tmp, path = self._candidate(first)
        self.addCleanup(tmp.cleanup)
        before = _candidate_execution_revision(path)

        path.write_text(second)
        after = _candidate_execution_revision(path)

        self.assertEqual(before["structure_sha256"], after["structure_sha256"])
        self.assertEqual(before["search_space_sha256"], after["search_space_sha256"])
        self.assertNotEqual(before["search_space_keys"], after["search_space_keys"])
        self.assertNotEqual(before["revision_sha256"], after["revision_sha256"])

    def test_stale_bytecode_and_mid_load_rewrite_are_refused(self) -> None:
        """A candidate rewritten under the loader must not be scored as this one."""
        tmp, path = self._candidate(self.SOURCE % 3)
        self.addCleanup(tmp.cleanup)
        (path.parent / "prepare.py").write_text(
            """
from pathlib import Path
candidate = Path(__file__).with_name("train.py")
candidate.write_text(
    candidate.read_text().replace("return params", "return 999")
)
""".lstrip()
        )

        with self.assertRaisesRegex(
            RuntimeError, "changed while modules were being loaded"
        ):
            load_candidate_modules(path)


class SearchSpaceBoundaryTests(unittest.TestCase):
    def test_categorical_values_are_primitive_finite_unique_and_type_exact(
        self,
    ) -> None:
        self.assertFalse(
            _valid_schema_entry(("categorical", [[1, 2]]))
        )
        self.assertFalse(
            valid_space_entry(("categorical", [1, 1.0]))
        )
        self.assertFalse(
            valid_space_entry(("categorical", [float("inf")]))
        )
        self.assertFalse(
            _schema_accepts_value(("categorical", [True]), 1)
        )
        verdict = check_search_space(
            {"flag": ("categorical", [True, False])},
            {"flag": ["categorical", [True, False]]},
            [{"flag": 1}],
        )
        self.assertFalse(verdict["ok"])
        self.assertIn(
            "config_value_invalid",
            {error["code"] for error in verdict["errors"]},
        )

    def test_malformed_space_and_extreme_numeric_return_errors(self) -> None:
        malformed = check_search_space(
            {"x": "int"},
            {"x": None},
            [{"x": 1}],
        )
        huge = check_search_space(
            {"x": "float"},
            {"x": ["float", 0.0, 1.0]},
            [{"x": 10**10000}],
        )

        self.assertFalse(malformed["ok"])
        self.assertIn(
            "bad_tuple",
            {error["code"] for error in malformed["errors"]},
        )
        self.assertFalse(huge["ok"])
        self.assertIn(
            "config_value_invalid",
            {error["code"] for error in huge["errors"]},
        )

    def test_configs_are_validated_against_complete_schema(self) -> None:
        proposed = {
            "depth": ["int", 1, 5],
            "rate": ["float", 0.001, 1.0, "log"],
            "mode": ["categorical", ["allowed"]],
        }
        invalid = [
            {"depth": "3", "rate": 0.1, "mode": "allowed"},
            {"depth": 3, "rate": 0.0, "mode": "allowed"},
            {"depth": 3, "rate": 0.1, "mode": "forbidden"},
            {"depth": 3, "rate": 0.1},
        ]

        verdict = check_search_space(SCHEMA, proposed, invalid)

        self.assertFalse(verdict["ok"])
        self.assertIsNone(verdict["finalized_space"])
        self.assertIn(
            "config_value_invalid",
            {error["code"] for error in verdict["errors"]},
        )
        self.assertIn(
            "config_key_mismatch",
            {error["code"] for error in verdict["errors"]},
        )

    def test_proposed_categorical_options_cannot_escape_schema(self) -> None:
        proposed = {
            "depth": ["int", 1, 5],
            "rate": ["float", 0.001, 1.0, "log"],
            "mode": ["categorical", ["allowed", "forbidden"]],
        }
        configs = [{"depth": 3, "rate": 0.1, "mode": "allowed"}]

        verdict = check_search_space(SCHEMA, proposed, configs)

        self.assertFalse(verdict["ok"])
        self.assertIn(
            "schema_mismatch",
            {error["code"] for error in verdict["errors"]},
        )

    def test_expansion_is_permutation_invariant(self) -> None:
        numeric_results = {
            repr(_expand_space_entry(("float", 0.0, 1.0), list(values)))
            for values in itertools.permutations([-10.0, 10.0, 0.5])
        }
        self.assertEqual(len(numeric_results), 1)

        log_results = {
            repr(_expand_space_entry(("float", 0.01, 1.0, "log"), list(values)))
            for values in itertools.permutations([0.001, 10.0, 0.1])
        }
        self.assertEqual(len(log_results), 1)
        log_space, _ = _expand_space_entry(
            ("float", 0.01, 1.0, "log"),
            [0.001, 10.0, 0.1],
        )
        self.assertGreater(log_space[1], 0)
        self.assertLess(log_space[1], 0.001)
        self.assertGreater(log_space[2], 10.0)

    def test_categorical_expansion_uses_schema_order_not_config_order(self) -> None:
        proposed = {
            "depth": ["int", 1, 5],
            "rate": ["float", 0.001, 1.0, "log"],
            "mode": ["categorical", ["allowed"]],
        }
        forward = [
            {"depth": 3, "rate": 0.1, "mode": "other"},
            {"depth": 4, "rate": 0.2, "mode": "allowed"},
        ]

        one = check_search_space(SCHEMA, proposed, forward)
        two = check_search_space(SCHEMA, proposed, list(reversed(forward)))

        self.assertTrue(one["ok"])
        self.assertEqual(one, two)
        self.assertEqual(
            one["finalized_space"]["mode"],
            ["categorical", ["allowed", "other"]],
        )


if __name__ == "__main__":
    unittest.main()
