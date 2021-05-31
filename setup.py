import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="panelpandas", # Replace with your own username
    version="0.0.1",
    author="Nicolas Woloszko",
    author_email="nicolas.woloszko@oecd.org",
    description="Tools for panel data in pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://algobank.oecd.org:4430/Nicolas.WOLOSZKO/panelpandas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.1',
)