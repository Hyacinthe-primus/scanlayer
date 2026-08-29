"""
CLI entry point for scanlayer.

Exit codes: 0=success, 1=user error, 2=env error, 3=unexpected,
4=processing error, 5=partial batch failure.
"""

from scanlayer.cli.run import main

__all__ = ["main"]
