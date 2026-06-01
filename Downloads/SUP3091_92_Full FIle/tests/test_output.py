import subprocess
import sys
import os
import pytest

BASE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')
)
PNG_PATH = os.path.join(BASE_DIR, 'SUP3091_3092_Timeline.png')
PDF_PATH = os.path.join(BASE_DIR, 'SUP3091_3092_Timeline.pdf')


@pytest.fixture(scope='module', autouse=True)
def run_script():
    """Run generate_timeline.py once before output tests."""
    result = subprocess.run(
        [sys.executable, 'generate_timeline.py'],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, \
        f"generate_timeline.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_png_exists():
    assert os.path.exists(PNG_PATH), f"PNG not found at {PNG_PATH}"


def test_pdf_exists():
    assert os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}"


def test_png_is_substantial():
    size = os.path.getsize(PNG_PATH)
    assert size > 500_000, f"PNG suspiciously small ({size} bytes) — rendering may have failed"


def test_pdf_is_substantial():
    size = os.path.getsize(PDF_PATH)
    assert size > 50_000, f"PDF suspiciously small ({size} bytes) — rendering may have failed"
