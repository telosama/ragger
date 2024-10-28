from setuptools import setup, find_packages

setup(
    name='my_awesome_package',
    version='0.1',
    packages=find_packages(),
    package_data={
        'my_package': ['*.py', '!old.py']
    },
    entry_points={
        'console_scripts': [
            'ragger = my_package.ragger:main'
        ],
    },
    install_requires=[
        'bs4'
        'gradio'
        'huggingface_hub'
        'langchain'
        'langchain-chroma'
        'langchain-community'
        'langchain-openai'
        'langgraph'
        'openai'
        'pypdf==5.0.1'
        'termcolor'
        'tiktoken'
    ],
)
