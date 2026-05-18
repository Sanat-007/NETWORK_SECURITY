from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements.
    """
    requirment_list:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            ##Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirment=line.strip()
                ## ignore empty lines and -e .
                if requirment and requirment!= '-e .':
                    requirment_list.append(requirment)

    except FileNotFoundError:
        print("requirements.txt file is not found.")

    return requirment_list

setup(
    name="NETWORK_SECURITY",
    version="0.1.0",
    author="Sanat",
    author_email="sanat18102007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)