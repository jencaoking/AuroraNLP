from setuptools import setup, find_packages

# 读取 README 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="AuroraNLP",
    version="0.3.0b1",
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    package_data={
        'AuroraNLP': ['data/*.txt', 'data/*.dict', 'data/domain_dictionaries/*.txt'],
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
    author="AuroraNLP Team",
    author_email="contact@auroranlp.example",
    description="AuroraNLP - 专业级中文自然语言处理工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="nlp segmentation chinese tokenizer ner pos-tagging",
    url="https://github.com/AuroraNLP/AuroraNLP",
    project_urls={
        "Bug Reports": "https://github.com/AuroraNLP/AuroraNLP/issues",
        "Source": "https://github.com/AuroraNLP/AuroraNLP",
        "Documentation": "https://github.com/AuroraNLP/AuroraNLP/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
