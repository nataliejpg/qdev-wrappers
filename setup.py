from setuptools import setup, find_packages

setup(
    name='qdev_wrappers',
    version='0.1',
    description='wrappers for helping to run an experiment with QCoDeS',
    url='https://github.com/qdev-dk/qdev-wrappers',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Licence :: MIT Licence',
        'Topic :: Scientific/Engineering'
    ],
    license='MIT',
    packages=find_packages(),
    package_data={'qcodes': ['config/*.json']},
    install_requires=[
        'matplotlib>=2.0.2',
        'pyqtgraph>=0.10.0',
        'qcodes>=0.1.9',
        'PyYAML>=3.12'
    ],
    python_requires='>=3'
)
