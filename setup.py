from setuptools import setup, find_packages

setup(
    name="pose_solver",
    version="1.0.0",
    description="Liquid Neural Network based Pose Solver",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8"
)