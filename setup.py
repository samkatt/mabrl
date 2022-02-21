#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "online_pomdp_planning @ git+ssh://git@github.com/samkatt/online-pomdp-planning.git@master",
    "pomdp_belief_tracking @ git+ssh://git@github.com/samkatt/pomdp-belief-tracking.git@main",
    "general_bayes_adaptive_pomdps @ git+ssh://git@github.com/samkatt/private-gbapomdp.git@cb95226443dbec5e31ddb1ee2cce7607e96c0fe8",
    "numpy",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="sammie katt",
    author_email="sammie.katt@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Bayes-adaptive approach towards solving multi-agent RL problems",
    entry_points={
        "console_scripts": [
            "mabrl=mabrl.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="mabrl",
    name="mabrl",
    packages=find_packages(include=["mabrl", "mabrl.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/samkatt/mabrl",
    version="0.1.0",
    zip_safe=False,
)
