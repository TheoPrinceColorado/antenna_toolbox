import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "antenna_toolbox",
    version = "0.0.6",
    author = "Theodore Prince & Jake Sahli",
    # author_email=,
    description = "A package for manipulation of antenna patterns",
    #long_description = long_description,
    #long_description_content_type = "text/markdown",
    url = "https://github.com/TheoPrinceColorado/antenna_toolbox",
    project_urls = {
        "Bug Tracker": "https://github.com/TheoPrinceColorado/antenna_toolbox/issues",
        "repository" :'https://github.com/TheoPrinceColorado/antenna_toolbox'
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "."},
    packages = setuptools.find_packages(exclude=['tests*']),
    python_requires = ">=3.8",
    install_requires=[
        'matplotlib>=3.3.2',
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'scipy>=1.5.3',
        'xarray>=2023.1.0',
    ]
)