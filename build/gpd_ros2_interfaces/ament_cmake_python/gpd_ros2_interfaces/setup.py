from setuptools import find_packages
from setuptools import setup

setup(
    name='gpd_ros2_interfaces',
    version='0.1.0',
    packages=find_packages(
        include=('gpd_ros2_interfaces', 'gpd_ros2_interfaces.*')),
)
