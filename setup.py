from setuptools import setup, find_packages

setup(
    name="ChemBioHepatox",
    version="0.1.0",
    author="Yingqing Shou, et al.",
    author_email="yfdeng@seu.edu.cn",
    description="A framework for hepatotoxicity prediction using chemical structure and biological fingerprints",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shouyqddd123/ChemBioHepatox",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "flask>=2.0.0",
        "tqdm>=4.60.0",
    ],
)
