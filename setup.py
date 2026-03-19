from setuptools import setup

packages = [
    'gb_code',
]

INSTALL_REQUIRES = [
    'numpy >= 1.14.0',
    'pyyaml',
]

EXTRAS_REQUIRE = {
    'pymatgen': ['pymatgen'],
    'ase': ['ase'],
    'all': ['pymatgen', 'ase'],
    'test': ['pytest', 'pymatgen', 'ase'],
}

setup(
    name='lz_GB_code',
    python_requires='>=3.9',
    version='0.1.0',
    author='R.Hadian',
    author_email='shahrzadhadian@gmail.com',
    maintainer='ligerzero-ai',
    packages=packages,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'csl_generator = gb_code.csl_generator:main',
            'gb_generator = gb_code.gb_generator:main',
            'inplane_shift = gb_code.inplane_shift:main',
        ],
    },
    url='https://github.com/ligerzero-ai/GB_code',
    license='MIT',
    description='Grain boundary generation code with pymatgen/ASE output support (fork of GB_code)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
