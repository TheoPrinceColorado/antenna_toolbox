import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "antenna_toolbox",
    version = "0.0.1",
    author = "Theodore Prince & Jake Sahli",
    # author_email=,
    description = "A package for manipulation of antenna patterns",
    #long_description = long_description,
    #long_description_content_type = "text/markdown",
    url = "https://github.com/TheoPrinceColorado/antenna_toolbox",
    # project_urls = {
    #     "Bug Tracker": "package issues URL",
    # },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "antenna_toolbox"},
    packages = setuptools.find_packages(where="antenna_toolbox"),
    python_requires = ">=3.6"
)