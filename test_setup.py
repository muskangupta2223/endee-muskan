#!/usr/bin/env python3
"""
System diagnostics script — verifies all components before running the app.

Checks (in order):
  0. Python version       — must be 3.8+
  1. Dependencies         — all pip packages installed
  2. CSV data file        — cleaned_tickets.csv present and valid
  3. Endee database       — server running at localhost:8080
  4. Embedding model      — all-MiniLM-L6-v2 loads and produces vectors
  5. Vector index         — 'tickets' index exists and returns search results
  6. Ollama LLM           — optional, needed for /assign-rag and /resolve-rag
  7. FastAPI server       — optional, checks if uvicorn is already running
  8. File structure       — all required project files are present

Usage:
    python test_setup.py
    python test_setup.py --verbose   # show debug logs
"""

import sys
import subprocess
import requests
import logging
import argparse
from pathlib import Path


class Colors:
    GREEN  = '\033[92m'
    RED    = '\033[91m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    END    = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'═' * 60}\n{text}\n{'═' * 60}{Colors.END}")

def print_success(text): print(f"{Colors.GREEN}✅ {text}{Colors.END}")
def print_error(text):   print(f"{Colors.RED}❌ {text}{Colors.END}")
def print_warning(text): print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")
def print_info(text, indent=1): print(f"{Colors.DIM}{'   ' * indent}{text}{Colors.END}")
def print_cmd(text):     print(f"{Colors.CYAN}   $ {text}{Colors.END}")


# --- Individual Test Functions ---

def test_python_version():
    print_header("0. Python Version")
    v = sys.version_info
    if v.major == 3 and v.minor >= 8:
        print_success(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    print_error(f"Python {v.major}.{v.minor}.{v.micro} — requires 3.8+")
    return False


def test_dependencies():
    """Check that all required packages can be imported."""
    print_header("1. Python Dependencies")
    packages = {
        'fastapi': 'fastapi', 'uvicorn': 'uvicorn', 'pydantic': 'pydantic',
        'sentence_transformers': 'sentence-transformers', 'requests': 'requests',
        'msgpack': 'msgpack', 'orjson': 'orjson', 'numpy': 'numpy',
        'pandas': 'pandas', 'tqdm': 'tqdm',
    }
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if not missing:
        print_success(f"All {len(packages)} packages installed")
        return True
    print_error(f"Missing: {', '.join(missing)}")
    print_cmd("pip install -r requirements.txt")
    return False


def test_csv_file():
    """Verify the CSV exists and has the correct columns."""
    print_header("2. Data File")
    csv_path = Path("data/cleaned_tickets.csv")
    if not csv_path.exists():
        print_error(f"Not found: {csv_path.absolute()}")
        print_info("Place cleaned_tickets.csv in the data/ directory")
        return False

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print_success(f"CSV found ({size_mb:.2f} MB)")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=5)
        required = ["ticket_id", "description", "team", "resolution"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print_warning(f"Missing columns: {missing}")
            return False
        print_info(f"Columns: {', '.join(df.columns)}")
    except Exception as e:
        print_warning(f"Could not validate structure: {e}")
    return True


def test_endee():
    """Check that Endee is running and responsive."""
    print_header("3. Endee Vector Database")
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        if r.status_code == 200:
            print_success("Endee running on port 8080")
            # Also show how many indexes currently exist
            try:
                from backend.endee_client import list_indexes
                idxs = list_indexes()
                print_info(f"{len(idxs)} index(es) found" if idxs else "No indexes yet — run ingest_tickets.py")
            except Exception:
                pass
            return True
        print_error(f"Endee returned status {r.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to http://localhost:8080")
        print_info("Start Endee:")
        print_cmd("cd ~/endee && export NDD_DATA_DIR=$(pwd)/data && ./build/ndd-avx2")
        return False


def test_embedding_model():
    """Load the model and verify it produces vectors of the correct dimension."""
    print_header("4. Embedding Model")
    try:
        from backend.embedder import embed_text, get_embedding_dimension
        print_info("Loading model (may take a few seconds on first run)...")
        vec = embed_text("test ticket description")
        dim = get_embedding_dimension()
        if len(vec) == dim:
            print_success(f"all-MiniLM-L6-v2 working (dim={dim})")
            return True
        print_error(f"Unexpected vector size: {len(vec)} (expected {dim})")
        return False
    except Exception as e:
        print_error(f"Failed: {e}")
        print_cmd("pip install sentence-transformers")
        return False


def test_vector_index():
    """Verify the 'tickets' index exists and returns results for a test query."""
    print_header("5. Vector Index Data")
    try:
        from backend.embedder import embed_text
        from backend.endee_client import search, get_index_info, INDEX_NAME

        # Show index metadata (dimension, count, etc.)
        try:
            info = get_index_info()
            print_info(f"Index '{INDEX_NAME}' found")
            if isinstance(info, dict):
                for k, v in info.items():
                    print_info(f"  {k}: {v}", indent=2)
        except Exception as e:
            print_warning(f"Could not get index info: {e}")

        # Run a test search to confirm data is actually in the index
        vec = embed_text("payment issue with credit card")
        matches = search(vec, top_k=1).get("results", [])

        if matches:
            print_success(f"Index has data ({len(matches)} result)")
            meta = matches[0].get("metadata", {})
            print_info(f"  Team: {meta.get('team', 'N/A')}  Score: {matches[0].get('score', 'N/A')}", indent=2)
            return True

        print_error("Index is empty — run ingest_tickets.py")
        return False

    except RuntimeError as e:
        # get_index_info raises a descriptive RuntimeError if index doesn't exist
        print_error(str(e))
        print_cmd("python ingest_tickets.py")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False


def test_ollama():
    """Check if Ollama is installed and the llama3 model is available."""
    print_header("6. Ollama LLM (Optional)")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5, text=True)
        if result.returncode != 0:
            print_error(f"Ollama failed: {result.stderr.strip()}")
            return False
        print_success("Ollama running")
        if "llama3" in result.stdout.lower():
            print_success("llama3 available — /assign-rag and /resolve-rag will work")
            return True
        print_warning("llama3 not found — pull it to enable RAG endpoints")
        print_cmd("ollama pull llama3")
        return False
    except FileNotFoundError:
        # Ollama isn't installed — non-critical, basic endpoints still work
        print_warning("Ollama not installed — /assign-rag and /resolve-rag won't work")
        print_info("Install: https://ollama.com/download")
        return False
    except Exception as e:
        print_error(str(e))
        return False


def test_api_server():
    """Check if the FastAPI server is already running."""
    print_header("7. FastAPI Server (Optional)")
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code in [200, 503]:
            print_success("API running on port 8000")
            for k, v in r.json().items():
                print_info(f"  {'✓' if v in [True, 'healthy'] else '✗'} {k}: {v}", indent=2)
            print_info("UI:   http://127.0.0.1:8000/ui")
            print_info("Docs: http://127.0.0.1:8000/docs")
            return True
        print_error(f"API returned {r.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print_warning("API not running — start it when ready:")
        print_cmd("uvicorn backend.main:app --reload")
        return False


def test_file_structure():
    """Verify all required project files and directories are present."""
    print_header("8. Project Structure")
    required = {
        "backend/":             "backend package directory",
        "backend/embedder.py":  "embedding module",
        "backend/endee_client.py": "Endee HTTP client",
        "backend/main.py":      "FastAPI application",
        "static/":              "static files directory",
        "static/index.html":    "web UI",
        "ingest_tickets.py":    "data ingestion script",
        "requirements.txt":     "Python dependencies",
    }
    missing = [(p, d) for p, d in required.items() if not Path(p).exists()]
    if not missing:
        print_success(f"All {len(required)} files/directories present")
        return True
    print_error(f"Missing {len(missing)} file(s):")
    for path, desc in missing:
        print_info(f"- {path} ({desc})", indent=2)
    return False


# --- Test Runner ---

def main():
    parser = argparse.ArgumentParser(description='SmartSupport AI Diagnostics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)

    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}{Colors.CYAN}SmartSupport AI - System Diagnostics{Colors.END}")
    print("=" * 60)

    # critical=True tests must pass for the app to function
    # critical=False tests are optional features
    tests = [
        ("Python Version",   test_python_version,   True),
        ("Dependencies",     test_dependencies,     True),
        ("Data File",        test_csv_file,         True),
        ("Endee Database",   test_endee,            True),
        ("Embedding Model",  test_embedding_model,  True),
        ("Vector Index",     test_vector_index,     True),
        ("Ollama LLM",       test_ollama,           False),
        ("API Server",       test_api_server,       False),
        ("File Structure",   test_file_structure,   True),
    ]

    results = {}
    for name, fn, critical in tests:
        try:
            results[name] = {"passed": fn(), "critical": critical}
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
            sys.exit(130)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results[name] = {"passed": False, "critical": critical}

    # Print summary table
    print_header("Summary")
    crit_pass = crit_total = opt_pass = opt_total = 0

    for name, r in results.items():
        if r["critical"]:
            crit_total += 1
            if r["passed"]:
                crit_pass += 1
                print_success(f"{name} (critical)")
            else:
                print_error(f"{name} (critical)")
        else:
            opt_total += 1
            if r["passed"]:
                opt_pass += 1
                print_success(f"{name} (optional)")
            else:
                print_warning(f"{name} (optional)")

    print(f"\n{Colors.BOLD}Results: {crit_pass}/{crit_total} critical  |  {opt_pass}/{opt_total} optional{Colors.END}")
    print("=" * 60)

    if crit_pass == crit_total:
        label = "🎉 All systems ready!" if opt_pass == opt_total else "✓ Core systems ready!"
        print(f"{Colors.GREEN}{Colors.BOLD}{label}{Colors.END}")
        print_cmd("uvicorn backend.main:app --reload")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Fix the critical failures above before starting the server{Colors.END}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
        sys.exit(130)