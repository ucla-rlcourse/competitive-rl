from setuptools import setup

setup(
    name="competitive_rl", version="0.0.1",
    packages=['competitive_rl', 'competitive_rl.pong'],
    install_requires=[
        "gym",
        "pygame",
        "opencv-python",
        "numpy",
        "pandas"
    ]
)
