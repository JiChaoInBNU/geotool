import sys
from setuptools import find_packages, setup

install_requires = [
    'numpy>=1.11.1',
    'GDAL',
]

setup(
    name='geotool',
    version='0.1.0',
    description='a tool implementation on gdal',
    keywords='geo tool python',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    url='https://github.com/Kindron/geotool.git',
    author='Kingdrone',
    author_email='kingdrone@163.com',
    license='GPLv3',
    setup_requires=[],
    tests_require=[],
    install_requires=install_requires,
    zip_safe=False)