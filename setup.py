from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="axiomatic-intelligence",
    version="1.0.0",
    author="Universal Standard Axiom Corp",
    author_email="research@universalstandardaxiom.com",
    description="Artificial General Intelligence via Axiomatic Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/universal-standard-axiom/axiomatic-intelligence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "myst-parser>=2.0.0",
        ],
    },
)
