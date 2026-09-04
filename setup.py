from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="moe-slice",
    version="0.1.0",
    author="ThaiNQ",
    author_email="thaicbhp5@gmail.com",
    description="A High-Performance Toolkit for Slicing, Profiling, and Calibrating Deep Sparse MoE LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jab1718/Moe-slices",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "safetensors>=0.4.0",
        "accelerate>=0.28.0"
    ],
    entry_points={
        "console_scripts": [
            "moe-slice=moe_slice.cli:main",
        ],
    },
)
