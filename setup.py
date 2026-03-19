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
    name='GB_code',
    python_requires='>3.5.1',
    version='2.0.0',
    author='R.Hadian',
    author_email='shahrzadhadian@gmail.com',
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
    url='https://github.com/oekosheri/GB_code',
    license='LICENSE',
    description='A GB generation code with pymatgen/ASE output support',
    long_description=open('README.md').read(),
)
