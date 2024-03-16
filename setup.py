from setuptools import setup, find_packages

setup(
    name='tm2048',
    version='0.1.0',
    url='https://github.com/lab2048/tm2048',
    author='Jilung Hsieh',
    author_email='jirlong@gmail.com',
    description='Tools for my lab(2048)',
    package_data={'tm2048': ['data/stopwords_cn.txt', 'data/stopwords_tw.txt']},
    include_package_data=True,
    packages=find_packages(),    
    install_requires=[
        'bokeh>=3.0',
        'pandas>=2.0',
        'gensim>=4.0',
    ]
)