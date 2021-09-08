import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="panelpandas", 
    version="0.0.1",
    author="Nicolas Woloszko",
    author_email="nicolas.woloszko@oecd.org",
    description="Tools for panel data in pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NicolasWoloszko/PanelData",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.1',
)