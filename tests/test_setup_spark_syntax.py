import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_setup_spark_script_is_syntactically_valid():
    script = REPO / "scripts" / "setup_spark.sh"
    assert script.exists(), "setup_spark.sh must exist"
    bash = shutil.which("bash")
    assert bash, "bash not on PATH"
    r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_setup_spark_is_executable():
    script = REPO / "scripts" / "setup_spark.sh"
    assert script.stat().st_mode & 0o111, "setup_spark.sh must have executable bit set"
