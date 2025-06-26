"""
Setup script for acne_diffusion package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="acne-diffusion",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning framework for acne image generation and classification using diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/acne-diffusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "wandb": ["wandb>=0.13.0"],
        "tensorboard": ["tensorboard>=2.10.0"],
    },
    entry_points={
        "console_scripts": [
            "acne-train-diffusion=scripts.train_diffusion:main",
            "acne-train-classifier=scripts.train_classifier:main",
            "acne-generate=scripts.generate_samples:main",
            "acne-evaluate=scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)