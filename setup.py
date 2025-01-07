from setuptools import setup, find_packages

setup(
    name="focus-mechanism",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
    ],
    author="Shiva Negi",
    author_email="your.email@example.com",
    description="A novel attention mechanism that combines traditional attention with dynamic Gaussian focus",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/focus-mechanism",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
