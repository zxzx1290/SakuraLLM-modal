# SakuraLLM-modal

A cloud-based batch translation pipeline powered by [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) and [Modal](https://modal.com). Translate `.txt` files at scale using GPU inference on the cloud — no local GPU required.

## Features

- Translate single `.txt` files or entire directories in one command
- Choose from multiple GPU tiers (L4 → H200) to balance cost and speed
- Two model options: a lightweight 1.5B model or a high-quality 7B model
- Automatic model caching on a persistent Modal volume — no redundant downloads
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

### Non-interactive / CLI mode

```bash
python modal_infer.py /path/to/file.txt --gpu L4 --model sakura-1.5b --non-interactive
```

**Arguments**

| Argument | Description | Default |
|---|---|---|
| `PATH` | Path to a `.txt` file or directory | *(required)* |
| `--gpu` | GPU type | `L4` |
| `--model` | `sakura-1.5b` or `galtransl-7b` | `sakura-1.5b` |
| `--text-length` | Max characters per inference chunk | `512` |
| `--timeout` | Task timeout in minutes | `240` |
| `--non-interactive` | Skip the confirmation prompt at the end | *(flag)* |

## Models

| Key | Model | Size | Notes |
|---|---|---|---|
| `sakura-1.5b` | Sakura-1.5B-Qwen2.5-v1.0 | 3.56 GB (FP16 GGUF) | Fast, lower resource usage |
| `galtransl-7b` | Sakura-GalTransl-7B-v3.7 | 6.34 GB (Q6_K GGUF) | Higher quality output |

## GPU Options

`L4` · `L40S` · `A10G` · `A100-40GB` · `A100-80GB` · `H100` · `H200` · `B200`

## Output

For each input file `<name>.txt`, the translated result is saved as `<name>_translated.txt` in the same directory. Execution logs are written to the `logs/` directory.

## How It Works

1. A unique session ID is generated for each run.
2. Input files are uploaded to a Modal persistent volume (`SakuraLLM_Data`).
3. A containerized function runs on the selected cloud GPU, clones the SakuraLLM repo, downloads the model (cached after first run), and performs translation.
4. The translated output is downloaded back to your local machine.
5. The cloud session directory is cleaned up automatically.

## Modal Secret Setup

Create a HuggingFace token secret in Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=<your_huggingface_token>
```

The token is required to download models from HuggingFace Hub.
