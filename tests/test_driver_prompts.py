import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.roles import PROMPT_DIR, ROLES  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_every_role_prompt_exists(self) -> None:
        for role in ROLES.values():
            self.assertTrue((PROMPT_DIR / role.prompt_file).exists(),
                            role.prompt_file)

    def test_no_frontmatter_remains(self) -> None:
        for role in ROLES.values():
            text = (PROMPT_DIR / role.prompt_file).read_text(encoding="utf-8")
            self.assertFalse(text.startswith("---\nname:"),
                             role.prompt_file)

    def test_every_prompt_names_the_receipt_tool(self) -> None:
        for role in ROLES.values():
            text = (PROMPT_DIR / role.prompt_file).read_text(encoding="utf-8")
            self.assertIn("mcp__receipts__submit_receipt", text,
                          role.prompt_file)

    def test_no_retired_paths_or_spawn_language(self) -> None:
        banned = re.compile(r"\.claude/(agents|skills|rules)|Skill\(crash-diagnosis\)")
        for role in ROLES.values():
            text = (PROMPT_DIR / role.prompt_file).read_text(encoding="utf-8")
            self.assertIsNone(banned.search(text), role.prompt_file)

    def test_extractor_prompt_cedes_resolve_unevaluated(self) -> None:
        text = (PROMPT_DIR / "tunable-contract-extractor.md").read_text(
            encoding="utf-8")
        self.assertNotIn("resolve-unevaluated", text)

    def test_rules_file_exists(self) -> None:
        self.assertTrue((PROMPT_DIR / "rules" / "ledger.md").exists())


if __name__ == "__main__":
    unittest.main()
