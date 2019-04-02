import os
from setuptools import find_packages, setup


about = {}
exec(open("appias/__version__.py").read(), about)

setup(
    name = "appias",
    version = about['__version__'],
    author = "Aaron Glover",
    author_email = "aglove2189@gmail.com",
    description = ("Machine learning workflow toolkit"),
    long_description = open("README.md").read(),
    license = "GPL",
    keywords = ["machine learning", "data science", "pandas", "sklearn"],
    url = "https://github.com/aglove2189/appias",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    platforms = ['Linux', 'OS-X', 'Windows'],
    packages = find_packages(exclude=('tests',)),
    install_requires = open("requirements.txt").read().split()
)
