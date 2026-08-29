"""
Entry point for `python -m scanlayer`.

Avoids double-importing main.py since __init__.py already imports it.
"""

from scanlayer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
