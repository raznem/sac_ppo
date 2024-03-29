import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("version.txt", "r") as f:
    version = f.read()

setuptools.setup(
    name="rltoolkit",
    version=version,
    author="MIMUWRL",
    author_email="",
    description="RL toolkit for experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIMUW-RL/a2c",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "PyVirtualDisplay==0.2.5",
        "gym==0.15.4",
        "torch>=1.3.1",
        "numpy>=1.18.1",
        "tensorboard",
    ],
)
