from setuptools import setup

setup(
    name="competitive_rl", version="0.1.0",
    packages=['competitive_rl', 'competitive_rl.pong', 'competitive_rl.car_racing'],
    install_requires=[
        "gym",
        "pygame==1.9.6",
        "opencv-python",
        "numpy",
        "pandas",
        "pyyaml",
        "box2d-py~=2.3.5"  # Identical to gym[box2d]
    ]
)
