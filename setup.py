import setuptools

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
    install_requires = [
        'pandas', 'numpy', 'scipy', 'gpflow', 'tensorflow', 'goatools', 'sklearn', 'statsmodels',
    ],
    include_package_data = True,
    package_data = {
        '': ['.csv', '.txt', '.xlsx']
    }
)