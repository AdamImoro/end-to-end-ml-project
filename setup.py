from setuptools import setup, find_packages
from typing import List

requirements =[]
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    """

    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]
    return requirements

setup(
    name='end-to-mlproject',
    version='0.0.1',
    author='Imoro',
    author_email='imoroadam895@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)