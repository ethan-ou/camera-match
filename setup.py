#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


requirements = ["numpy", "scipy", "matplotlib", "colour-science>=0.3.16"]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Ethan Ou",
    author_email="ethantim@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="Match two cameras together using multiple algorithms",
    install_requires=requirements,
    extras_require={
        "RBF": ["xalglib"],
    },
    license="MIT license",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="camera_match",
    name="camera_match",
    packages=find_packages(include=["camera_match", "camera_match.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ethan-ou/camera_match",
    version="0.0.3",
    zip_safe=False,
)
