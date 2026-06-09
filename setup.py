# ============================================================
# SETUP FILE (EDITABLE INSTALL CONFIG)
# ------------------------------------------------------------
# Purpose:
# This file allows the project to be installed as a Python
# package in editable mode using:
#
#     pip install -e .
#
# This ensures that the "src" folder is globally importable
# without sys.path hacks or notebook dependencies.
# ============================================================

from setuptools import setup, find_packages

setup(
    name="nlp-sentiment-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[]
)