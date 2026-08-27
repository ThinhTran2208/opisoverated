"""Lock the lightweight cross-platform PR checks requested for portability."""

import unittest
from pathlib import Path


class PortabilityCiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.workflow = (
            repo_root / ".github/workflows/portability-tests.yml"
        ).read_text(encoding="utf-8")

    def test_ci_covers_requested_operating_systems_and_python_versions(self):
        for required_value in (
            "ubuntu-latest",
            "windows-latest",
            '"3.10"',
            '"3.11"',
        ):
            self.assertIn(required_value, self.workflow)

    def test_ci_runs_only_lightweight_static_and_unit_checks(self):
        self.assertIn("python -m compileall -q src tests", self.workflow)
        self.assertIn(
            "python -m unittest discover -s tests -v", self.workflow
        )
        self.assertNotIn("NB2_core7", self.workflow)
        self.assertNotIn("NB3_core7", self.workflow)
        self.assertNotIn("NB4_build", self.workflow)


if __name__ == "__main__":
    unittest.main()
