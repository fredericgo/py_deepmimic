import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_deepmimic",
    version="0.0.1",
    author="Frederic Go",
    author_email="fredericstgo@gmail.com",
    description="a deepmimic implementation",
    long_description=long_description,
    url="https://github.com/fredericgo/py_deepmimic",
    packages=setuptools.find_packages(include=["py_deepmimic.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
