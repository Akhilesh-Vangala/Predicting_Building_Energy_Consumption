from setuptools import find_packages, setup

setup(
    name="bep",
    version="0.1.0",
    description="Predicting Building Energy Consumption — DS-GA 1003 final project",
    author="Akhilesh Vangala, Lucas Yao, Anvita Reddy",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
