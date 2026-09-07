"""Settings must load the service's .env no matter where the process was started."""

import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_env_file_resolves_to_service_root():
    from src.core.config import ENV_FILE

    assert ENV_FILE == SERVICE_ROOT / ".env"
    assert ENV_FILE.is_absolute()


def test_settings_load_from_a_foreign_working_directory(tmp_path):
    """Regression: uvicorn started from frontend/ read the wrong .env, so every
    LLM call failed with "No LLM provider credentials configured"."""
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.core.config import settings, ENV_FILE\n"
        "print(ENV_FILE)\n"
        "print(settings.database_url)\n" % str(SERVICE_ROOT)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    env_file, database_url = result.stdout.strip().splitlines()
    assert env_file == str(SERVICE_ROOT / ".env")
    assert database_url
