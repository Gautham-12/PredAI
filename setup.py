"""Setup configuration for PredAI Predictive Maintenance Platform (Emperor Syndicate)."""
from setuptools import setup, find_packages

setup(
    name="predai",
    version="1.0.0",
    description="PredAI: Autonomous Predictive Maintenance Platform for Turbofan Engines",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Emperor Syndicate / PredAI Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "lightgbm>=4.0",
    ],
    extras_require={
        "deep_learning": ["torch>=2.0"],
        "dashboard": ["streamlit>=1.30", "plotly>=5.0"],
        "mlops": ["mlflow>=2.10"],
        "validation": ["pydantic>=2.0"],
        "dev": ["pytest>=8.0"],
        "all": [
            "torch>=2.0",
            "streamlit>=1.30",
            "plotly>=5.0",
            "mlflow>=2.10",
            "pydantic>=2.0",
            "pytest>=8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "predai-train=pipelines.train_pipeline:main",
            "predai-infer=pipelines.inference_pipeline:main",
            "aura-train=pipelines.train_pipeline:main",
            "aura-infer=pipelines.inference_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
