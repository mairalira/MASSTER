import os
from setuptools import setup, find_packages
from pathlib import Path


def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pyclus',
    python_requires='>=3.6',
    version='1.0.2',
    description='Wrapper around Clus.',
    author='Matej Petkovic, Martin Breskvar',
    author_email='015280203a@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    long_description=long_description,
    long_description_content_type='text/markdown'
)
