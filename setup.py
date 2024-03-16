from setuptools import setup, find_packages

setup(
    name='lib2048',
    version='0.1.0',
    url='https://github.com/lab2048/lib2048',
    author='Jilung Hsieh',
    author_email='jirlong@gmail.com',
    description='Tools for my lab(2048)',
    packages=find_packages(),    
    install_requires=[
        'pandas>=2.0',
    ],
)