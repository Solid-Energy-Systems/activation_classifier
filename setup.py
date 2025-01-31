from setuptools import setup, find_packages

setup(
    name="activation_classifier",
    version="0.1.0",
    description="A Python package for predicting properties of text from LLM activations.",
    author="Adam Atanas",
    author_email="adam.atanas@ses.ai",
    url="https://github.com/Solid-Energy-Systems/activation_classifier",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        'torch',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)

