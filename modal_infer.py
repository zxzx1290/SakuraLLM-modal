from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

def ensure_utf8_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            encoding = getattr(stream, "encoding", None)
            if encoding and encoding.lower().startswith("utf-8"):
                continue
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            elif hasattr(stream, "buffer"):
                setattr(
                    sys,
                    name,
                    io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"),
                )
        except Exception:
            pass

ensure_utf8_stdio()

try:
    import questionary
    from questionary import Choice
except ImportError:
    questionary = None
    Choice = None

try:
    import modal
except ImportError:
    print("未偵測到 modal 套件，請先執行 `python -m pip install modal questionary`。")
    raise

APP_NAME = "SakuraLLM-Translate"
REPO_URL = "https://github.com/SakuraLLM/SakuraLLM"
VOLUME_NAME = "SakuraLLM_Data"
VOLUME_ROOT = "/SakuraLLM_Data"
REMOTE_MOUNT = VOLUME_ROOT
SESSION_SUBDIR = "sessions"
REPO_VOLUME_DIR = f"{VOLUME_ROOT}/repo"
TXT_SUFFIXES = {".txt"}

DEFAULT_GPU_CHOICES = [
    "T4",
    "L4",
    "L40S",
    "A10G",
    "A100-40GB",
    "A100-80GB",
    "H100",
    "H200",
    "B200",
]


@dataclass
class ModelProfile:
    key: str
    label: str
    hf_repo: str | None
    description: str
    gguf_file: str | None = None  # GGUF 檔名（若為 GGUF 模型）
    model_version: str = "0.10"


@dataclass
class UserSelection:
    gpu_choice: str
    input_path: Path
    model_profile: ModelProfile
    text_length: int
    timeout_minutes: int


@dataclass
class UploadManifest:
    session_id: str
    source_type: str  # file or directory
    local_source: Path
    remote_inputs_rel: list[Path]
    remote_output_rel: Path
    local_output_dir: Path
    original_filename: str | None = None


MODEL_PRESETS: dict[str, ModelProfile] = {
    "sakura-1.5b": ModelProfile(
        key="sakura-1.5b",
        label="Sakura-1.5B-Qwen2.5-v1.0",
        hf_repo="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
        description="1.5B GGUF FP16 3.56GB",
        gguf_file="sakura-1.5b-qwen2.5-v1.0-fp16.gguf",
        model_version="0.10",
    ),
    "galtransl-7b": ModelProfile(
        key="galtransl-7b",
        label="Sakura-GalTransl-7B-v3.7",
        hf_repo="SakuraLLM/Sakura-GalTransl-7B-v3.7",
        description="7B GGUF Q6_K 6.34GB",
        gguf_file="Sakura-Galtransl-7B-v3.7.gguf",
        model_version="0.10",
    ),
}


def rel_to_volume_path(path: Path) -> str:
    posix = path.as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix
    return posix


def rel_to_container_path(path: Path) -> str:
    base = PurePosixPath(REMOTE_MOUNT)
    return str((base / path.as_posix()).as_posix())


def setup_logger() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modal_run_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info("日誌輸出：%s", log_path)
    return log_path


def ensure_questionary():
    if questionary is None or Choice is None:
        raise RuntimeError("需要 questionary，請執行 `python -m pip install questionary`。")


def ask_selection(args: argparse.Namespace) -> UserSelection:
    # 有 --input 參數時直接使用 CLI 參數，否則進入互動模式
    if args.input:
        model_profile = MODEL_PRESETS[args.model]
        input_path = Path(args.input.strip().strip("'\"")).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"路徑不存在：{input_path}")
        return UserSelection(
            gpu_choice=args.gpu,
            input_path=input_path,
            model_profile=model_profile,
            text_length=args.text_length,
            timeout_minutes=args.timeout,
        )

    # 互動模式
    ensure_questionary()

    gpu_choice = questionary.select(
        "選擇 GPU",
        choices=DEFAULT_GPU_CHOICES,
        default=DEFAULT_GPU_CHOICES[0],
    ).ask()
    if not gpu_choice:
        raise KeyboardInterrupt

    model_key = questionary.select(
        "選擇模型：",
        choices=[Choice(title=f"{p.label} - {p.description}", value=k) for k, p in MODEL_PRESETS.items()],
        default=next(iter(MODEL_PRESETS)),
    ).ask()
    if not model_key:
        raise KeyboardInterrupt

    model_profile = MODEL_PRESETS[model_key]

    input_path_str = questionary.path("拖入或輸入待翻譯的 txt 檔案/資料夾路徑：").ask()
    if not input_path_str:
        raise KeyboardInterrupt
    input_path = Path(input_path_str.strip().strip("'\"")).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"路徑不存在：{input_path}")

    text_length = int(questionary.text("每次推理的最大文字長度", default=str(args.text_length)).ask() or str(args.text_length))
    timeout_minutes = int(questionary.text("任務逾時時間（分鐘）", default=str(args.timeout)).ask() or str(args.timeout))

    return UserSelection(
        gpu_choice=gpu_choice,
        input_path=input_path,
        model_profile=model_profile,
        text_length=text_length,
        timeout_minutes=timeout_minutes,
    )


def scan_txt_files(path: Path) -> list[Path]:
    """掃描資料夾中的 txt 檔案"""
    return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in TXT_SUFFIXES)


def validate_input_path(path: Path) -> list[Path]:
    """驗證輸入路徑，回傳 txt 檔案清單"""
    if path.is_file():
        if path.suffix.lower() not in TXT_SUFFIXES:
            raise ValueError(f"檔案 {path} 不是 txt 格式。")
        return [path]
    elif path.is_dir():
        files = scan_txt_files(path)
        if not files:
            raise FileNotFoundError(f"資料夾內沒有 txt 檔案：{path}")
        return files
    else:
        raise ValueError(f"路徑 {path} 既不是檔案也不是資料夾。")


def upload_single_file(
    volume: modal.Volume,
    audio_file: Path,
    base_dir: Path | None = None,
) -> UploadManifest:
    """上傳單個 txt 檔案到 Modal Volume"""
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    remote_session_rel = Path(SESSION_SUBDIR) / session_id

    original_filename = audio_file.name
    safe_filename = "input.txt"

    with volume.batch_upload(force=True) as batch:
        remote_rel = remote_session_rel / safe_filename
        logging.info("上傳檔案 -> %s", rel_to_volume_path(remote_rel))
        batch.put_file(str(audio_file), rel_to_volume_path(remote_rel))

    local_output_dir = base_dir if base_dir else audio_file.parent

    return UploadManifest(
        session_id=session_id,
        source_type="file",
        local_source=audio_file,
        remote_inputs_rel=[remote_rel],
        remote_output_rel=remote_session_rel,
        local_output_dir=local_output_dir,
        original_filename=original_filename,
    )


def build_job_payload(selection: UserSelection, manifest: UploadManifest) -> dict:
    model_profile = selection.model_profile

    return {
        "session_id": manifest.session_id,
        "mount_root": str(REMOTE_MOUNT),
        "repo_url": REPO_URL,
        "remote_input": rel_to_container_path(manifest.remote_inputs_rel[0]),
        "remote_output_dir": rel_to_container_path(manifest.remote_output_rel),
        "hf_repo": model_profile.hf_repo,
        "gguf_file": model_profile.gguf_file,
        "model_version": model_profile.model_version,
        "text_length": selection.text_length,
        "timeout_seconds": selection.timeout_minutes * 60,
    }


def _build_modal_image() -> modal.Image:
    # 動態讀取本機 Python 版本，確保雲端 image 與執行腳本的版本一致。
    # 版本一致後，serialized=True 的序列化才不會報錯（closure 內的 function 需要它）。
    # 跨電腦也相容：各電腦用自己的 Python 版本建立各自的 image。
    #
    # 取捨說明：
    #   - 換電腦且 Python 版本不同時，Modal 會視為不同 image 而重建（約數分鐘）。
    #   - 重建只發生一次，之後該版本的 image 就有快取，後續執行秒啟動。
    #   - 若想避免重建，可改為寫死版本（如 python_version="3.11"），
    #     但本機版本不同時 serialized=True 可能再度出現序列化相容問題。
    _py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        modal.Image.debian_slim(python_version=_py)
        .apt_install("git")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
        .pip_install(
            "transformers==4.38.0",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "tqdm",
            "dacite",
            "huggingface_hub",
            "pydantic",
            "coloredlogs",
            "opencc",
            "pysubs2",
            "scipy",
            "numpy",
            "modal",
        )
        .run_commands(
            "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir"
        )
    )


_modal_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def download_outputs(manifest: UploadManifest, result: dict) -> None:
    """從遠端結果取出翻譯檔案並寫入本地"""
    translated_content = result.get("translated_content")
    if not translated_content:
        logging.warning("未收到翻譯結果")
        return

    # 使用原始檔名加上 _translated 後綴
    original_stem = Path(manifest.original_filename).stem if manifest.original_filename else "input"
    output_filename = f"{original_stem}_translated.txt"

    local_path = manifest.local_output_dir / output_filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(translated_content)
    logging.info("寫入翻譯結果: %s (%d bytes)", local_path, len(translated_content))

    # 寫入 log
    log_content = result.get("log_content")
    if log_content:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"modal_run_{manifest.session_id}.log"
        log_path.write_bytes(log_content)
        logging.info("寫入日誌: %s", log_path)


def process_files(
    selection: UserSelection,
    txt_files: list[Path],
) -> tuple[int, int]:
    """處理所有 txt 檔案，容器復用"""
    logging.info("使用 GPU：%s", selection.gpu_choice)
    logging.info("使用模型：%s（%s）", selection.model_profile.label, selection.model_profile.description)
    logging.info("文字長度：%d", selection.text_length)
    logging.info("逾時時間：%d 分鐘", selection.timeout_minutes)
    logging.info("待處理檔案數：%d", len(txt_files))

    image = _build_modal_image()
    app = modal.App(APP_NAME)

    # modal_pipeline 刻意定義在此函式內部（巢狀函式），
    # 目的是讓 @app.function(gpu=...) 能在 runtime 接收使用者動態選擇的 GPU 型號。
    # Modal 的 GPU 設定必須在裝飾器宣告時指定，無法事後修改，
    # 因此只能透過巢狀函式在每次呼叫時重新定義。
    #
    # 副作用：巢狀函式需要 serialized=True 才能被 Modal 序列化傳送到雲端，
    # 而 serialized=True 要求本機與雲端的 Python 版本完全一致。
    # 這就是 _build_modal_image() 動態讀取本機版本的原因。
    @app.function(
        image=image,
        gpu=selection.gpu_choice,
        timeout=selection.timeout_minutes * 60,
        volumes={str(REMOTE_MOUNT): _modal_volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
        serialized=True,
        min_containers=1,
    )
    def modal_pipeline(job_payload: dict) -> dict:
        return _remote_pipeline(job_payload)

    success_count = 0
    fail_count = 0
    base_dir = selection.input_path if selection.input_path.is_dir() else None

    with app.run():
        for i, txt_file in enumerate(txt_files, 1):
            logging.info("=" * 60)
            logging.info("處理檔案 [%d/%d]: %s", i, len(txt_files), txt_file.name)
            logging.info("=" * 60)
            try:
                t_start = datetime.now()
                manifest = upload_single_file(_modal_volume, txt_file, base_dir)
                payload = build_job_payload(selection, manifest)
                logging.info("正在執行翻譯...")
                result = modal_pipeline.remote(payload)
                download_outputs(manifest, result)
                session_path = rel_to_volume_path(manifest.remote_output_rel)
                _modal_volume.remove_file(session_path, recursive=True)
                logging.info("已清除雲端 session 目錄: %s", session_path)
                elapsed = (datetime.now() - t_start).total_seconds()
                if elapsed < 60:
                    elapsed_str = f"{elapsed:.1f} 秒"
                elif elapsed < 3600:
                    elapsed_str = f"{int(elapsed // 60)} 分 {int(elapsed % 60)} 秒"
                else:
                    elapsed_str = f"{int(elapsed // 3600)} 小時 {int(elapsed % 3600 // 60)} 分 {int(elapsed % 60)} 秒"
                logging.info("檔案 %s 處理完成（耗時 %s）", txt_file.name, elapsed_str)
                success_count += 1
            except Exception as e:
                logging.error("檔案 %s 處理失敗: %s", txt_file.name, e)
                fail_count += 1
                continue

    return success_count, fail_count


def parse_args() -> argparse.Namespace:
    model_keys = list(MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(description="SakuraLLM Modal 翻譯腳本")
    parser.add_argument(
        "--gpu",
        choices=DEFAULT_GPU_CHOICES,
        default=DEFAULT_GPU_CHOICES[0],
        help=f"GPU 型號（預設 {DEFAULT_GPU_CHOICES[0]}，可選：{', '.join(DEFAULT_GPU_CHOICES)}）",
    )
    parser.add_argument(
        "--model",
        choices=model_keys,
        default=model_keys[0],
        help=f"模型（預設 {model_keys[0]}，可選：{', '.join(model_keys)}）",
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="待翻譯的 txt 檔案或資料夾路徑",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=512,
        metavar="N",
        help="每次推理的最大文字長度（預設 512）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        metavar="MINUTES",
        help="任務逾時時間，單位分鐘（預設 240）",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="執行完畢後不等待按鍵。",
    )
    return parser.parse_args()


def prompt_exit(enabled: bool) -> None:
    if not enabled:
        return
    with contextlib.suppress(EOFError):
        input("輸入任意鍵退出...")


def main() -> int:
    args = parse_args()
    log_path = setup_logger()
    exit_code = 0
    try:
        selection = ask_selection(args)

        txt_files = validate_input_path(selection.input_path)
        logging.info("共找到 %d 個 txt 檔案待翻譯", len(txt_files))

        success_count, fail_count = process_files(selection, txt_files)

        logging.info("=" * 60)
        logging.info("=== 翻譯完成 ===")
        logging.info("成功: %d, 失敗: %d", success_count, fail_count)
        if selection.input_path.is_dir():
            logging.info("輸出路徑: %s", selection.input_path)
        else:
            logging.info("輸出路徑: %s", selection.input_path.parent)
        logging.info("請在上方輸出路徑查看翻譯結果。")
    except KeyboardInterrupt:
        logging.warning("使用者中斷，未執行任何遠端操作。")
        exit_code = 1
    except Exception as exc:
        logging.exception("執行失敗：%s", exc)
        logging.error("日誌見：%s", log_path)
        exit_code = 1

    prompt_exit(not args.non_interactive)
    return exit_code


def _remote_pipeline(job: dict) -> dict:
    # 以下 import 刻意放在函式內，不可上拉到根層級。
    # 此函式會被 Modal 序列化後傳送到雲端 VM 執行，
    # 函式內的 import 會在雲端才觸發，確保使用雲端環境的套件版本。
    # 若放到根層級，會在本機執行時就 import，可能與雲端環境產生衝突。
    import os
    import subprocess
    import time
    from pathlib import Path

    from modal import Volume

    volume = Volume.from_name("SakuraLLM_Data")
    volume.reload()

    def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=cwd, env=env)

    mount_root = Path(job["mount_root"])
    repo_dir = Path(REPO_VOLUME_DIR)

    session_dir = Path(job["remote_output_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "modal_run.log"

    def log(msg: str) -> None:
        line = f"[sakura_modal] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # 1. 克隆或更新 SakuraLLM repo
    if not (repo_dir / ".git").exists():
        log("開始克隆 SakuraLLM 倉庫...")
        run(["git", "clone", "--depth", "1", job["repo_url"], str(repo_dir)])
    else:
        log("更新倉庫...")
        run(["git", "-C", str(repo_dir), "fetch", "origin"])
        run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"])

    # 2. 下載模型（如果尚未存在）
    hf_repo = job["hf_repo"]
    model_dir = mount_root / "models" / hf_repo.replace("/", "_")

    gguf_file = job["gguf_file"]
    gguf_path = model_dir / gguf_file

    if not gguf_path.exists():
        log(f"下載模型 {hf_repo} / {gguf_file}...")
        model_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=hf_repo,
            filename=gguf_file,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        volume.commit()
        log(f"模型下載完成: {gguf_path}")
    else:
        log(f"模型已存在: {gguf_path}")

    # 3. 等待輸入檔案就緒
    input_path = Path(job["remote_input"])
    log(f"等待輸入檔案: {input_path}")
    max_wait = 180
    waited = 0
    while not input_path.exists() and waited < max_wait:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            log(f"等待檔案出現... ({waited}s)")

    if not input_path.exists():
        raise FileNotFoundError(f"輸入檔案未出現: {input_path}")
    log(f"檔案已就緒: {input_path}")

    # 4. 執行翻譯
    output_path = session_dir / "output_translated.txt"

    cmd = [
        "python",
        str(repo_dir / "translate_novel.py"),
        "--model_name_or_path", str(gguf_path),
        "--model_version", str(job["model_version"]),
        "--llama_cpp",
        "--use_gpu",
        "--n_gpu_layers", "-1",
        "--data_path", str(input_path),
        "--output_path", str(output_path),
        "--text_length", str(job["text_length"]),
    ]

    log(f"執行翻譯命令：{' '.join(cmd)}")
    env = os.environ.copy()
    run(cmd, cwd=str(repo_dir), env=env)

    # 5. 讀取翻譯結果
    translated_content = None
    if output_path.exists():
        translated_content = output_path.read_bytes()
        log(f"翻譯完成，輸出大小: {output_path.stat().st_size} bytes")
    else:
        log("警告：未找到翻譯輸出檔案")

    # 6. 讀取 log
    log_content = None
    if log_file.exists():
        log_content = log_file.read_bytes()

    return {
        "translated_content": translated_content,
        "log_content": log_content,
    }


if __name__ == "__main__":
    sys.exit(main())
