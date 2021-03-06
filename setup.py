import setuptools
import os

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    requirements = []
else:
    requirements = [
        'pandas>=0.25.1', 
        'numpy>=1.16.5', 
        'scipy>=1.3.1', 
        'gpflow==1.5.1', 
        'tensorflow==1.15.2', 
        'goatools>=1.0.2', 
        'scikit-learn>=0.21.3', 
        'statsmodels>=0.10.1',
        'matplotlib>=3.1.1', 
        'seaborn>=0.9.0', 
        'pytest'
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TRANSPIRE", 
    version="0.1.1.dev1",
    author="Michelle A. Kennedy",
    author_email="mak4515@gmail.com",
    description="A Python package for TRanslocation ANalysis of SPatIal pRotEomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mak4515/TRANSPIRE",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = requirements,
    include_package_data = True,
    package_data = {
        '': ['.csv', '.txt', '.xlsx']
    }
)