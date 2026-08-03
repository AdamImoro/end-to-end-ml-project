from setuptools import find_packages, setup
from typing import List 

HYPHEN_DOT_E = "-e ."

requirements = []
def get_requirements(file_path:str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    """

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)

    return requirements


setup(
    name="End-to-End-ml",
    author='imoro',
    version="0.0.1",
    author_email="imoroadam895@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)