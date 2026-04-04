from setuptools import setup, find_packages

setup(
    name="nlp_segment",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'nlp_segment': ['data/*.txt'],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    author="NLP Team",
    description="NLP分词工具包",
    keywords="nlp segmentation chinese",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
