from setuptools import setup

setup(
    name="HealthSyn",
    version="0.1",
    description=open("README.md", "rt").read(),
    packages=[
        "syngen",
        "syngen.data",
    ],
    install_requires=[
        "numpy",
        "pydantic",
        "scipy",
        "tqdm",
        "typing_extensions",
    ],
)
