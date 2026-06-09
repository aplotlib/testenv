import subprocess
import sys
import os
import pytest

BASE_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
PDF_PATH  = os.path.join(BASE_DIR, 'SUP3091_3092_Quality_Analysis.pdf')
PNG_PATHS = [
    os.path.join(BASE_DIR, f'SUP3091_3092_Quality_Analysis_p{i}.png')
    for i in range(1, 5)
]


@pytest.fixture(scope='module', autouse=True)
def run_script():
    result = subprocess.run(
        [sys.executable, 'generate_quality_analysis.py'],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, \
        f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_pdf_exists():
    assert os.path.exists(PDF_PATH)


def test_pdf_is_substantial():
    assert os.path.getsize(PDF_PATH) > 100_000, \
        f"PDF too small: {os.path.getsize(PDF_PATH)} bytes"


def test_four_page_pngs_exist():
    for p in PNG_PATHS:
        assert os.path.exists(p), f"Missing: {p}"


def test_each_png_is_substantial():
    for p in PNG_PATHS:
        size = os.path.getsize(p)
        assert size > 50_000, f"PNG too small ({size} bytes): {p}"
