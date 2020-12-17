from setuptools import setup

setup(
    name="competitive_rl", version="0.1.0",
    packages=['competitive_rl', 'competitive_rl.pong', 'competitive_rl.car_racing', 'competitive_rl.utils'],
    install_requires=[
        "gym",
        "pygame==1.9.6",
        "opencv-python",
        "numpy",
        "pandas",
        "pyyaml",
        "tabulate",
        "box2d-py~=2.3.5"  # Identical to gym[box2d]
    ],
    package_data={"competitive_rl": ["car_racing/fonts/*", "pong/*.ttf", "car_racing/*.png"]}
)

print("You have successfully installed competitive_rl 0.1.0!")
