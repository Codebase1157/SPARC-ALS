"""setup.py — editable install for development."""
from setuptools import find_packages, setup

setup(
    name="sparc-als",
    version="1.0.0",
    description="Speaker-Invariant Pathology-Aware Representation with Vowel-Space Graph Attention for Cross-Corpus ALS Detection",
    author="Komal Ahuja, Amanpreet Kaur",
    author_email="komal1034cse.phd24@chitkara.edu.in",
    license="MIT",
    url="https://github.com/komal1034/sparc-als",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    python_requires=">=3.10,<3.12",
    include_package_data=True,
)
