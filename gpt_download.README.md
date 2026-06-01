# gpt_download.py

This file has been removed from this repository to comply with licensing requirements.

## Original Source

The original `gpt_download.py` file (authored by Sebastian Raschka) is available at:

```
https://raw.githubusercontent.com/rickiepark/llm-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py
```

## License

The original code is licensed under the Apache License 2.0.
See the `LICENSE` file in this repository for details.

## How to Use

You can download and use the original file directly:

```python
import urllib.request

url = (
    "https://raw.githubusercontent.com/rickiepark/"
    "llm-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)

filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

# Then import and use
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

Alternatively, the function is already included in the Jupyter notebooks (e.g., ch05.ipynb) via the same URL import pattern.
