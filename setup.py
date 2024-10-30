from setuptools import find_packages, setup

setup(
    name="ELPF",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        # Add any other dependencies your project has
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Expected Likelihood Particle Filter for state estimation.",
    license="MIT",
    keywords="particle filter state estimation",
)
