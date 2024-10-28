Ragger
======
```ragger.py``` is a command line tool to RAG over multiple pieces of text, PDFs and websites using langchain and OpenAI.

The main repository is on ![git.telosama.xyz](https://git.telosama.xyz/iang/ragger), where issues and contributions are accepted

## Prerequisites
Since this codebase uses OpenAI models, a OpenAI API key is needed.
After obtaining an API key, set the environment variable OPENAI_API_KEY to the API key.

## Installation
This program can be installed by cloning the repository and installing the package using pip:
```bash
git clone gitea@git.telosama.xyz:iang/ragger.git
cd ragger
pip install .
```

## Usage
To use the `ragger` command, open a terminal and run:
```bash
ragger file_1_path.pdf file_2_path.txt https://url_1_path.net
```