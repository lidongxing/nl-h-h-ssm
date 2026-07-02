from setuptools import find_namespace_packages, setup

setup(
    name="nlh-ssm",
    version="0.1.0",
    description="NL-H-H-SSM",
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "pyyaml>=6.0",
        "pandas>=1.5,<2.3",
        # NumPy 2.x can break torch / mamba_ssm wheels built against NumPy 1.x ABI.
        "numpy>=1.22,<2",
    ],
    packages=find_namespace_packages(
        include=[
            "nlh_ssm",
            "nlh_ssm.*",
            "csrc",
            "models",
            "models.*",
            "benchmarks",
            "benchmarks.*",
            "experiments",
            "experiments.*",
        ]
    ),
    extras_require={
        # mamba_ssm SSD ops need triton.language.cumsum (Triton 2.1+); align with torch's supported triton.
        "mamba": ["mamba-ssm", "triton>=2.1.0"],
    },
)
