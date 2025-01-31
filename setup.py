from setuptools import setup, find_packages

setup(
    name="dinamo",
    version="1.0.0",
    author="Arsenii Gavrikov, Julián García Pardiñas",
    author_email="agavriko@cern.ch, julian.garcia.pardinas@cern.ch",
    packages=find_packages(),
    package_dir={"dinamo": "dinamo"},
    install_requires=[
        "matplotlib==3.7.2",
        "numpy==1.26.3",
        "optuna==3.5.0",
        "pandas==2.1.1",
        "scikit-learn==1.2.2",
        "scipy==1.11.4",
        "seaborn==0.13.2",
        "setuptools==68.2.2",
        "torch==2.4.0",
        "tqdm==4.65.0",
    ],
)
