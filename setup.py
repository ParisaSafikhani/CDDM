from setuptools import setup, find_packages

setup(
    name="cddm",
    version="0.1.0",
    description="Corpus-Driven Domain Mapping",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Parisa Safikhani",
    author_email="parisa.safikhani@ovgu.de",
    url="https://github.com/ParisaSafikhani/CDDM",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.22.0",
        "sentence-transformers>=2.0.0",
        "huggingface_hub>=0.0.19",
        "transformers>=4.0.0",
        "tqdm>=4.45.0",
        "torch>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
) 