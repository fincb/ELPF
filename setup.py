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
        'plotly'
    ],
    author="Joshua J Wakefield",
    author_email="sgjwakef@liverpool.ac.uk",
    description="Expected Likelihood Particle Filter for state estimationin cluttered environments.",
    license="MIT",
    keywords="particle filter state estimation",
)
