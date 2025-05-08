import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'autodrive_race'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'autodrive_race'), glob('autodrive_race/*.csv')),
        (os.path.join('share', package_name, 'autodrive_race'), glob('autodrive_race/*.py')),
        (os.path.join('share', package_name, 'checkpoints'), glob('checkpoints/*')),
        (os.path.join('share', package_name, 'best_model'), glob('best_model/*')),
        (os.path.join('share', package_name, 'logs'), glob('logs/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*'))  
          
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ppo_training=autodrive_race.ppo_training:main',
            'eval=autodrive_race.eval:main',
            'final_test_training=autodrive_race.final_test_training:main'
        ],
    },
)
