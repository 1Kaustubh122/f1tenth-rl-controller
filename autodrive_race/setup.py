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
            # 'ips_to_tf_brodcaster= autodrive_mapping.ips_to_tf_brodcaster:main',
            # 'record_waypoints=autodrive_mapping.record_waypoints:main',
            # 'cimppc_node=autodrive_mapping.cimppc_node:main',
            # 'pure_pursuit_node=autodrive_mapping.PurePursuit:main',
            # 'gen_traj=autodrive_mapping.gen_traj:main',
            'qual_round=autodrive_race.qual_round:main'
        ],
    },
)
