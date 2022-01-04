#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Ethan Ou",
    author_email='ethantim@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Match two cameras together using multiple algorithms",
    entry_points={
        'console_scripts': [
            'camera_match=camera_match.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='camera_match',
    name='camera_match',
    packages=find_packages(include=['camera_match', 'camera_match.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ethan-ou/camera_match',
    version='0.1.0',
    zip_safe=False,
)
