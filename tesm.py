#!/usr/bin/env python3

from ragger import parse_markdown

markdown_text = """
# Header 1
## h2
### h3
Some regular text with **bold** and *italic* and `inline code`.
another text with *italic and **bold** text*

## **GPT style headers**:

> This is a blockquote.

## Header 2

1. First item
2. Second item
- Bullet item 1
* Bullet item 2

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| Row 1 | Data | Info |
| Row 2 | More | Details |

```python
# This is a code block
print("Hello, world!")
```
"""

parsed_output = parse_markdown(markdown_text)
print(parsed_output)