# SakuraLLM-modal

A cloud-based batch translation pipeline powered by [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) and [Modal](https://modal.com). Translate `.txt` files at scale using GPU inference on the cloud — no local GPU required.

## Features

- Translate single `.txt` files or entire directories in one command
- Choose from multiple GPU tiers (L4 → H200) to balance cost and speed
- Multiple model options: 1.5B, 7B, and 14B
- Automatic model caching on a persistent Modal volume — no redundant downloads
- Custom glossary dictionary support — auto-detected as `gpt_dict.txt` or specified via `--dict`
- Interactive CLI or fully scriptable non-interactive mode
- Per-file error isolation — a failed file won't stop the rest of the batch

## Requirements

- Python 3.x
- A [Modal](https://modal.com) account with the CLI configured (`modal setup`)
- A HuggingFace token stored as a Modal secret named `huggingface-secret`

## Installation

```bash
pip install modal questionary
```

## Usage

### Interactive mode

```bash
python modal_infer.py
```

You will be prompted to select:
- GPU type
- Model
- Input file or directory path
- Text length per inference (default: 512)
- Timeout in minutes (default: 240)
- Glossary dictionary file path (optional)

### Non-interactive / CLI mode

```bash
python modal_infer.py /path/to/file.txt --gpu L4 --model sakura-1.5b --non-interactive
```

With a glossary dictionary:

```bash
python modal_infer.py /path/to/file.txt --dict my_dict.txt
```

**Arguments**

| Argument | Description | Default |
|---|---|---|
| `PATH` | Path to a `.txt` file or directory | *(required)* |
| `--gpu` | GPU type | `L4` |
| `--model` | `sakura-14b-q6k`, `sakura-1.5b`, or `galtransl-7b` | `sakura-14b-q6k` |
| `--dict` | Path to a glossary dictionary `.txt` file | *(none)* |
| `--text-length` | Max characters per inference chunk | `512` |
| `--timeout` | Task timeout in minutes | `240` |
| `--non-interactive` | Skip the confirmation prompt at the end | *(flag)* |

### Glossary dictionary format

One entry per line. The format matches the official SakuraLLM convention:

```
原文->譯文
原文->譯文#備註
```

Lines starting with `#` are treated as comments and ignored.

Example:

```
# Character names
アリア->艾莉亞#主角名
魔王城->魔王城堡
```

You can generate a dictionary from an existing JSON source using the official [`convert_to_gpt_dict.py`](https://github.com/SakuraLLM/SakuraLLM/blob/main/convert_to_gpt_dict.py) tool.

### Auto-detection

If a file named `gpt_dict.txt` exists in the same directory as the input file (or the input directory itself), it is loaded automatically — no `--dict` flag needed. Explicitly passing `--dict` always takes priority.

## Models

| Key | Model | Size | Notes |
|---|---|---|---|
| `sakura-14b-q6k` | Sakura-14B-Qwen3-v1.5 (Q6_K) | 12.1 GB | Best quality, requires A10G or higher |
| `sakura-1.5b` | Sakura-1.5B-Qwen2.5-v1.0 | 3.56 GB (FP16 GGUF) | Fast, lower resource usage |
| `galtransl-7b` | Sakura-GalTransl-7B-v3.7 | 6.34 GB (Q6_K GGUF) | Higher quality, visual novel focus |

## GPU Options

`L4` · `L40S` · `A10G` · `A100-40GB` · `A100-80GB` · `H100` · `H200` · `B200`

## Output

For each input file `<name>.txt`, the translated result is saved as `<name>_translated.txt` in the same directory. Execution logs are written to the `logs/` directory.

## How It Works

1. A unique session ID is generated for each run.
2. Input files (and optional dictionary) are uploaded to a Modal persistent volume (`SakuraLLM_Data`).
3. A containerized function runs on the selected cloud GPU, downloads the model (cached after first run), and performs translation using the bundled scripts.
4. The translated output is downloaded back to your local machine.
5. The cloud session directory is cleaned up automatically.

> **Note:** This project bundles a modified `translate_novel.py`, `sampler_hijack.py`, `utils/`, and `infers/` from [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) (GPL v3). The main modification is the addition of `--gpt_dict_path` support in `translate_novel.py`.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

The following files are copied from [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) (GPL v3) and included in this repository under the same license:
- `translate_novel.py` — modified to add `--gpt_dict_path` argument and `load_gpt_dict()`
- `sampler_hijack.py`, `utils/`, `infers/` — unmodified

## Modal Secret Setup

Create a HuggingFace token secret in Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=<your_huggingface_token>
```

The token is required to download models from HuggingFace Hub.
