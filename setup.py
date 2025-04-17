from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bacsc",
    version="0.1.0",
    author="johannesostner",
    description="A general workflow for bacterial single-cell RNA sequencing data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bio-datascience/BacSC",
    packages=["bacsc", "bacsc.tools"],
    package_dir={"bacsc": ".", "bacsc.tools": "tools"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
    install_requires=[
        "scanpy",
        "KDEpy",
        "leidenalg",
        "venn",
        "scipy",
        "statsmodels",
        "pandas",
        "numpy",
        "anndata",
        "seaborn",
        "matplotlib",
        "setuptools",
    ],
)
