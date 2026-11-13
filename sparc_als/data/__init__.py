"""Data-handling subpackage.

Submodules are imported lazily so that `preprocess.py --dry-run` works
without torch installed.
"""
__all__ = ["preprocess", "forced_alignment", "splits", "loaders"]
