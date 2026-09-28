# Copyright 2026 Mahmud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

package_name = 'semantic_nav_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    extras_require={
        'test': [
            'pytest',
        ],
    },
    maintainer='Mahmud',
    maintainer_email='mahmudlawal75@gmail.com',
    description='YOLO object detection and 3D projection of detections into the map frame.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detector_node = semantic_nav_perception.detector_node:main',
            'projector_node = semantic_nav_perception.projector_node:main',
        ],
    },
)
