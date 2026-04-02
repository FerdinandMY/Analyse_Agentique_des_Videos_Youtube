from setuptools import find_packages, setup


setup(
    name="agentic_video_analysis",
    version="0.1.0",
    description="Agentic multi-agent pipeline for video analysis (scaffold).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "setuptools>=68.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.10",
)