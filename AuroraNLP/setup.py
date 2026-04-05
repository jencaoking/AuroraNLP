from setuptools import setup, find_packages

setup(
    name="AuroraNLP",
    version="0.3.0b1",
    codename="coca",
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'AuroraNLP': ['data/*.txt'],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "lint": [
            "mypy>=1.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
        ],
    },
    author="NLP Team",
    description="AuroraNLP - 中文自然语言处理工具包",
    keywords="nlp segmentation chinese",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
