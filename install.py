"""Install the exact runtime dependencies, including PyTDC without extras."""
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC==1.1.15", "--no-deps"])
