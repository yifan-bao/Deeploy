import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DumpO",
    version="0.0.1",
    author="Moritz Scherer",
    author_email="scheremo@iis.ee.ethz.ch",
    description="DumpO - Bring you networks to your platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://iis-git.ee.ethz.ch/scheremo/dumpo",
    project_urls={
        "Bug Tracker": "https://iis-git.ee.ethz.ch/scheremo/dumpo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"DumpO": "DumpO"},
    packages=['DumpO'] + ['.'.join(['DumpO', p]) for p in setuptools.find_packages(os.path.curdir + '/DumpO')],
    python_requires=">=3.6",
)
