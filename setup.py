#!/usr/bin/env python3
"""
setup.py
─────────
One-time setup script: creates virtual env, installs dependencies,
downloads NLP models, initialises the database, and verifies the installation.

Usage:
    python setup.py
    python setup.py --skip-venv      # if already in a venv
    python setup.py --light          # skip heavy torch/transformers install
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DATA = ROOT / "data"


def run(cmd: list, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def python_bin():
    if VENV.exists():
        if sys.platform == "win32":
            return str(VENV / "Scripts" / "python.exe")
        return str(VENV / "bin" / "python")
    return sys.executable


def pip_bin():
    if VENV.exists():
        if sys.platform == "win32":
            return str(VENV / "Scripts" / "pip.exe")
        return str(VENV / "bin" / "pip")
    return str(Path(sys.executable).parent / "pip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-venv", action="store_true", help="Skip venv creation")
    parser.add_argument("--light",     action="store_true", help="Skip torch/transformers (lighter install)")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  AI Text Humanizer — Setup")
    print("═" * 55)

    # ── 1. Create virtual environment ─────────────────────────────────────────
    if not args.skip_venv:
        if VENV.exists():
            print(f"\n✓ Virtual environment already exists at {VENV}")
        else:
            print(f"\n→ Creating virtual environment at {VENV}")
            run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print("\n→ Skipping venv creation (--skip-venv)")

    py = python_bin()
    pip = pip_bin()

    # ── 2. Upgrade pip ────────────────────────────────────────────────────────
    print("\n→ Upgrading pip")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # ── 3. Install requirements ───────────────────────────────────────────────
    req_file = ROOT / "requirements.txt"
    if args.light:
        print("\n→ Installing lightweight requirements (skipping torch)")
        # Filter out heavy packages
        light_req = ROOT / "requirements_light.txt"
        with open(req_file) as f:
            lines = [l for l in f if not any(
                pkg in l.lower() for pkg in ["torch", "transformers", "sentence"]
            )]
        light_req.write_text("".join(lines))
        run([py, "-m", "pip", "install", "-r", str(light_req)])
        light_req.unlink()
    else:
        print("\n→ Installing all requirements")
        run([py, "-m", "pip", "install", "-r", str(req_file)])

    # ── 4. Download spaCy model ───────────────────────────────────────────────
    print("\n→ Downloading spaCy English model (en_core_web_sm)")
    run([py, "-m", "spacy", "download", "en_core_web_sm"])

    # ── 5. Download NLTK data ─────────────────────────────────────────────────
    print("\n→ Downloading NLTK punkt tokeniser")
    run([py, "-c",
         "import nltk; nltk.download('punkt', quiet=True); "
         "nltk.download('stopwords', quiet=True); "
         "nltk.download('averaged_perceptron_tagger', quiet=True)"])

    # ── 6. Create .env from example ──────────────────────────────────────────
    env_file    = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"\n✓ Created .env from .env.example — edit it to add your API keys.")
    elif not env_file.exists():
        env_file.write_text(
            "# Add your API key here\n"
            "OPENAI_API_KEY=sk-...\n"
            "LLM_PROVIDER=openai\n"
        )
        print(f"\n✓ Created minimal .env — add your API key.")
    else:
        print(f"\n✓ .env already exists.")

    # ── 7. Create data directories ────────────────────────────────────────────
    for d in [DATA, DATA / "uploads"]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory: {d}")

    # ── 8. Verify ─────────────────────────────────────────────────────────────
    print("\n→ Verifying installation…")
    verify_script = """
import sys
errors = []
try:    import fastapi; print("  ✓ fastapi", fastapi.__version__)
except: errors.append("fastapi")
try:    import spacy; spacy.load('en_core_web_sm'); print("  ✓ spacy + en_core_web_sm")
except: errors.append("spacy/en_core_web_sm")
try:    import nltk; print("  ✓ nltk", nltk.__version__)
except: errors.append("nltk")
try:    import textstat; print("  ✓ textstat")
except: errors.append("textstat")
try:    import sqlalchemy; print("  ✓ sqlalchemy", sqlalchemy.__version__)
except: errors.append("sqlalchemy")
try:    import aiosqlite; print("  ✓ aiosqlite")
except: errors.append("aiosqlite")
try:    import docx; print("  ✓ python-docx")
except: errors.append("python-docx")
if errors:
    print(f"\\n  MISSING: {errors}")
    sys.exit(1)
else:
    print("\\n  All core dependencies OK!")
"""
    run([py, "-c", verify_script])

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Setup complete! To start the server:")
    if VENV.exists():
        activate = ".venv\\Scripts\\activate" if sys.platform == "win32" else "source .venv/bin/activate"
        print(f"\n    {activate}")
    print("    python run.py")
    print(f"\n  Then open:  http://127.0.0.1:8000")
    print(f"  API docs:   http://127.0.0.1:8000/docs")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
