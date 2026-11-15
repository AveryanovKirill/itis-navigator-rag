"""Usage: python eval/run.py --attacks --backend live --output reports/security_live_01"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main

if __name__ == '__main__':
    sys.exit(main())
