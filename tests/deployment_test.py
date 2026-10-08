"""Exercise the production script's activation and rollback against fake services.

Run on Linux with GNU coreutils: python3 tests/deployment_test.py
Dependency installation/builds are excluded; symlink and trap behavior is real.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "scripts/deploy.sh").read_text()
SHA = "a" * 40


class DeploymentTest(unittest.TestCase):
    def run_deploy(self, failure):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for directory in ("releases", "old-release", "old-runtime", "commands"):
                (base / directory).mkdir()
            (base / "current").symlink_to(base / "old-release")
            (base / "venv").symlink_to(base / "old-runtime")
            commands = {
                "sudo": '''#!/bin/bash
if [[ "$FAILURE" == signal && ! -f "$BASE/signalled" ]]; then
  touch "$BASE/signalled"
  kill -TERM "$PPID"
fi
exit 0
''',
                "curl": '''#!/bin/bash
if [[ "$FAILURE" == health && ! -f "$BASE/failed" && "$(readlink -f "$BASE/current")" != "$BASE/old-release" ]]; then
  touch "$BASE/failed"
  exit 22
fi
exit 0
''',
                "git": '#!/bin/bash\nprintf "%s\\n" "$SHA"\n',
            }
            for name, content in commands.items():
                path = base / "commands" / name
                path.write_text(content)
                path.chmod(0o700)
            prefix, preparation = SOURCE.split('git clone --quiet', 1)
            _, activation = preparation.split('activated=true\n', 1)
            script = prefix.replace('base=/srv/trust.tgo.dev', f'base={base}')
            script += 'mkdir -p "$release/.venv"\nactivated=true\n' + activation
            environment = dict(os.environ, FAILURE=failure, BASE=str(base), SHA=SHA)
            environment["PATH"] = str(base / "commands") + ":" + environment["PATH"]
            result = subprocess.run(
                ["bash", "-se", "--", SHA], input=script, text=True,
                capture_output=True, env=environment, timeout=10,
            )
            return result, (base / "current").resolve().name, (base / "venv").resolve().name

    def test_success_activates_matching_runtime(self):
        result, release, runtime = self.run_deploy("none")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(release.startswith(SHA[:12]))
        self.assertEqual(runtime, ".venv")

    def test_failed_health_restores_both_paths(self):
        result, release, runtime = self.run_deploy("health")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((release, runtime), ("old-release", "old-runtime"))
        self.assertIn("restored", result.stderr)

    def test_termination_restores_both_paths(self):
        result, release, runtime = self.run_deploy("signal")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertEqual((release, runtime), ("old-release", "old-runtime"))
        self.assertIn("restored", result.stderr)


if __name__ == "__main__":
    unittest.main()
