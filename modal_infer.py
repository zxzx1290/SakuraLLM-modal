from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import time
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
VOLUME_NAME = "SakuraLLM_Data"
VOLUME_ROOT = "/SakuraLLM_Data"
REMOTE_MOUNT = VOLUME_ROOT
SESSION_SUBDIR = "sessions"
SAKURA_APP_DIR = "/opt/sakura"  # image 內的翻譯腳本與 utils 目錄
TXT_SUFFIXES = {".txt"}

DEFAULT_GPU_CHOICES = [
    "T4",   # $0.59/hr，建議僅用於 7B 模型（VRAM 16GB）
    "L4",   # $0.80/hr
    "A10G", # $1.10/hr
    "L40S", # $1.95/hr
]


@dataclass
class ModelProfile:
    key: str
    label: str
    hf_repo: str | None
    description: str
    gguf_file: str | None = None  # GGUF 檔名（若為 GGUF 模型）
    model_version: str = "0.10"
    default_temperature: float | None = None  # 模型建議的 temperature
    default_top_p: float | None = None  # 模型建議的 top-p


@dataclass
class UserSelection:
    gpu_choice: str
    input_path: Path
    model_profile: ModelProfile
    text_length: int
    timeout_minutes: int
    dict_path: Path | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class UploadManifest:
    session_id: str
    source_type: str  # file or directory
    local_source: Path
    remote_inputs_rel: list[Path]
    remote_output_rel: Path
    local_output_dir: Path
    original_filename: str | None = None
    remote_dict_rel: Path | None = None


MODEL_PRESETS: dict[str, ModelProfile] = {
    "galtransl-14b": ModelProfile(
        key="galtransl-14b",
        label="Sakura-GalTransl-14B-v3.8",
        hf_repo="SakuraLLM/Sakura-GalTransl-14B-v3.8",
        description="14B GGUF Q6_K 12.1GB",
        gguf_file="Sakura-Galtransl-14B-v3.8.gguf",
        model_version="0.10",
        default_temperature=0.3,
        default_top_p=0.8,
    ),
    "sakura-14b-q6k": ModelProfile(
        key="sakura-14b-q6k",
        label="Sakura-14B-Qwen3-v1.5 (Q6_K)",
        hf_repo="SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF",
        description="14B GGUF Q6_K 12.1GB",
        gguf_file="sakura-14b-qwen3-v1.5-q6k.gguf",
        model_version="0.10",
    ),
    "galtransl-7b": ModelProfile(
        key="galtransl-7b",
        label="Sakura-GalTransl-7B-v3.7",
        hf_repo="SakuraLLM/Sakura-GalTransl-7B-v3.7",
        description="7B GGUF Q6_K 6.34GB",
        gguf_file="Sakura-Galtransl-7B-v3.7.gguf",
        model_version="0.10",
        default_temperature=0.3,
        default_top_p=0.8,
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


AUTO_DICT_FILENAME = "gpt_dict.txt"


def resolve_dict_path(explicit: Path | None, input_path: Path) -> Path | None:
    """回傳最終使用的字典路徑：明確指定 > 輸入目錄 > 專案根目錄 > None"""
    if explicit is not None:
        return explicit
    _project_root = Path(__file__).parent
    candidates = [
        (input_path if input_path.is_dir() else input_path.parent) / AUTO_DICT_FILENAME,
        _project_root / AUTO_DICT_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            logging.info("自動偵測到字典檔：%s", candidate)
            validate_dict_file(candidate)
            return candidate
    return None


def ask_selection(args: argparse.Namespace) -> UserSelection:
    # 有路徑參數時直接使用 CLI 參數，否則進入互動模式
    if args.path:
        model_profile = MODEL_PRESETS[args.model]
        input_path = Path(args.path.strip().strip("'\"")).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"路徑不存在：{input_path}")
        explicit_dict: Path | None = None
        if args.dict:
            explicit_dict = Path(args.dict.strip().strip("'\"")).expanduser().resolve()
            if not explicit_dict.exists():
                raise FileNotFoundError(f"字典檔案不存在：{explicit_dict}")
            validate_dict_file(explicit_dict)
        dict_path = resolve_dict_path(explicit_dict, input_path)
        return UserSelection(
            gpu_choice=args.gpu,
            input_path=input_path,
            model_profile=model_profile,
            text_length=args.text_length,
            timeout_minutes=args.timeout,
            dict_path=dict_path,
            temperature=args.temperature,
            top_p=args.top_p,
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

    dict_path_str = questionary.text("（選填）自訂字典 txt 檔案路徑（格式：原文->譯文#備註），直接 Enter 跳過：", default="").ask()
    explicit_dict: Path | None = None
    if dict_path_str and dict_path_str.strip():
        explicit_dict = Path(dict_path_str.strip().strip("'\"")).expanduser().resolve()
        if not explicit_dict.exists():
            raise FileNotFoundError(f"字典檔案不存在：{explicit_dict}")
        validate_dict_file(explicit_dict)
    dict_path = resolve_dict_path(explicit_dict, input_path)

    default_temp = str(model_profile.default_temperature) if model_profile.default_temperature is not None else ""
    default_top_p = str(model_profile.default_top_p) if model_profile.default_top_p is not None else ""
    temperature_str = questionary.text(f"（選填）temperature，直接 Enter 使用模型預設值{f'（{default_temp}）' if default_temp else ''}：", default=default_temp).ask()
    top_p_str = questionary.text(f"（選填）top_p，直接 Enter 使用模型預設值{f'（{default_top_p}）' if default_top_p else ''}：", default=default_top_p).ask()
    temperature = float(temperature_str) if temperature_str and temperature_str.strip() else None
    top_p = float(top_p_str) if top_p_str and top_p_str.strip() else None

    return UserSelection(
        gpu_choice=gpu_choice,
        input_path=input_path,
        model_profile=model_profile,
        text_length=text_length,
        timeout_minutes=timeout_minutes,
        dict_path=dict_path,
        temperature=temperature,
        top_p=top_p,
    )


def scan_txt_files(path: Path) -> list[Path]:
    """掃描資料夾中的 txt 檔案"""
    return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in TXT_SUFFIXES)


def validate_dict_file(path: Path) -> None:
    """驗證字典檔格式，有問題直接 raise ValueError"""
    errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                errors.append(f"  第 {lineno} 行缺少 '->'：{line!r}")
            else:
                src, temp = line.split("->", 1)
                if not src.strip():
                    errors.append(f"  第 {lineno} 行原文為空：{line!r}")
                if not temp.split("#", 1)[0].strip():
                    errors.append(f"  第 {lineno} 行譯文為空：{line!r}")
    if errors:
        raise ValueError("字典檔格式錯誤：\n" + "\n".join(errors))


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
    dict_path: Path | None = None,
) -> UploadManifest:
    """上傳單個 txt 檔案（及可選的字典檔）到 Modal Volume"""
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    remote_session_rel = Path(SESSION_SUBDIR) / session_id

    original_filename = audio_file.name
    safe_filename = "input.txt"

    remote_dict_rel: Path | None = None

    with volume.batch_upload(force=True) as batch:
        remote_rel = remote_session_rel / safe_filename
        logging.info("上傳檔案 -> %s", rel_to_volume_path(remote_rel))
        batch.put_file(str(audio_file), rel_to_volume_path(remote_rel))

        if dict_path is not None:
            remote_dict_rel = remote_session_rel / "gpt_dict.json"
            logging.info("上傳字典 -> %s", rel_to_volume_path(remote_dict_rel))
            batch.put_file(str(dict_path), rel_to_volume_path(remote_dict_rel))

    local_output_dir = base_dir if base_dir else audio_file.parent

    return UploadManifest(
        session_id=session_id,
        source_type="file",
        local_source=audio_file,
        remote_inputs_rel=[remote_rel],
        remote_output_rel=remote_session_rel,
        local_output_dir=local_output_dir,
        original_filename=original_filename,
        remote_dict_rel=remote_dict_rel,
    )


def build_job_payload(selection: UserSelection, manifest: UploadManifest) -> dict:
    model_profile = selection.model_profile

    payload: dict = {
        "session_id": manifest.session_id,
        "mount_root": str(REMOTE_MOUNT),
        "remote_input": rel_to_container_path(manifest.remote_inputs_rel[0]),
        "remote_output_dir": rel_to_container_path(manifest.remote_output_rel),
        "hf_repo": model_profile.hf_repo,
        "gguf_file": model_profile.gguf_file,
        "model_version": model_profile.model_version,
        "text_length": selection.text_length,
        "timeout_seconds": selection.timeout_minutes * 60,
        "temperature": selection.temperature if selection.temperature is not None else model_profile.default_temperature,
        "top_p": selection.top_p if selection.top_p is not None else model_profile.default_top_p,
    }

    if manifest.remote_dict_rel is not None:
        payload["remote_dict"] = rel_to_container_path(manifest.remote_dict_rel)

    return payload


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
        .add_local_file("translate_novel.py", f"{SAKURA_APP_DIR}/translate_novel.py")
        .add_local_file("sampler_hijack.py", f"{SAKURA_APP_DIR}/sampler_hijack.py")
        .add_local_dir("utils", f"{SAKURA_APP_DIR}/utils")
        .add_local_dir("infers", f"{SAKURA_APP_DIR}/infers")
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
    effective_temperature = selection.temperature if selection.temperature is not None else selection.model_profile.default_temperature
    effective_top_p = selection.top_p if selection.top_p is not None else selection.model_profile.default_top_p
    logging.info("使用 GPU：%s", selection.gpu_choice)
    logging.info("使用模型：%s（%s）", selection.model_profile.label, selection.model_profile.description)
    logging.info("文字長度：%d", selection.text_length)
    logging.info("逾時時間：%d 分鐘", selection.timeout_minutes)
    logging.info("字典檔：%s", selection.dict_path if selection.dict_path is not None else "無")
    logging.info("temperature：%s", effective_temperature if effective_temperature is not None else "模型預設")
    logging.info("top_p：%s", effective_top_p if effective_top_p is not None else "模型預設")
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

    logging.info("3 秒後開始執行，按 Ctrl+C 可取消...")
    time.sleep(3)
    logging.info("開始執行 Modal 推理流程...")
    with app.run():
        for i, txt_file in enumerate(txt_files, 1):
            logging.info("=" * 60)
            logging.info("處理檔案 [%d/%d]: %s", i, len(txt_files), txt_file.name)
            logging.info("=" * 60)
            manifest: UploadManifest | None = None
            try:
                t_start = datetime.now()
                manifest = upload_single_file(_modal_volume, txt_file, base_dir, selection.dict_path)
                payload = build_job_payload(selection, manifest)
                logging.info("正在執行翻譯...")
                result = modal_pipeline.remote(payload)
                download_outputs(manifest, result)
                elapsed = (datetime.now() - t_start).total_seconds()
                if elapsed < 60:
                    elapsed_str = f"{elapsed:.1f} 秒"
                elif elapsed < 3600:
                    elapsed_str = f"{int(elapsed // 60)} 分 {int(elapsed % 60)} 秒"
                else:
                    elapsed_str = f"{int(elapsed // 3600)} 小時 {int(elapsed % 3600 // 60)} 分 {int(elapsed % 60)} 秒"
                elapsed_translate = result.get("elapsed_translate")
                output_chars = result.get("output_chars", 0)
                if elapsed_translate and elapsed_translate > 0 and output_chars:
                    chars_per_sec = output_chars / elapsed_translate
                    logging.info(
                        "推理速度：%.1f 字元/秒（翻譯耗時 %.1fs，輸出 %d 字元）",
                        chars_per_sec, elapsed_translate, output_chars,
                    )
                logging.info("檔案 %s 處理完成（耗時 %s）", txt_file.name, elapsed_str)
                success_count += 1
            except Exception as e:
                logging.error("檔案 %s 處理失敗: %s", txt_file.name, e)
                fail_count += 1

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
        "path",
        nargs="?",
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
        "--dict",
        metavar="DICT_PATH",
        default=None,
        help="自訂字典 txt 檔案路徑，每行格式：原文->譯文  或  原文->譯文#備註",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help="採樣溫度",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        default=None,
        metavar="P",
        help="Top-p 採樣",
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
    app_dir = Path(SAKURA_APP_DIR)

    session_dir = Path(job["remote_output_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "modal_run.log"

    def log(msg: str) -> None:
        line = f"[sakura_modal] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # 1. 下載模型（如果尚未存在）
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
        str(app_dir / "translate_novel.py"),
        "--model_name_or_path", str(gguf_path),
        "--model_version", str(job["model_version"]),
        "--llama_cpp",
        "--use_gpu",
        "--n_gpu_layers", "-1",
        "--data_path", str(input_path),
        "--output_path", str(output_path),
        "--text_length", str(job["text_length"]),
    ]

    remote_dict = job.get("remote_dict")
    if remote_dict:
        cmd += ["--gpt_dict_path", remote_dict]
        log(f"使用自訂字典：{remote_dict}")

    temperature = job.get("temperature")
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
        log(f"temperature：{temperature}")

    top_p = job.get("top_p")
    if top_p is not None:
        cmd += ["--top_p", str(top_p)]
        log(f"top_p：{top_p}")

    input_chars = len(input_path.read_text(encoding="utf-8", errors="replace"))

    log(f"執行翻譯命令：{' '.join(cmd)}")
    env = os.environ.copy()
    t_translate_start = time.time()
    run(cmd, cwd=str(app_dir), env=env)
    elapsed_translate = time.time() - t_translate_start

    # 5. 讀取翻譯結果
    translated_content = None
    output_chars = 0
    if output_path.exists():
        translated_content = output_path.read_bytes()
        output_chars = len(translated_content.decode("utf-8", errors="replace"))
        chars_per_sec = output_chars / elapsed_translate if elapsed_translate > 0 else 0
        log(
            f"翻譯完成，耗時 {elapsed_translate:.1f}s，"
            f"輸入 {input_chars} 字元，輸出 {output_chars} 字元，"
            f"速度 {chars_per_sec:.1f} 字元/秒"
        )
    else:
        log("警告：未找到翻譯輸出檔案")

    # 6. 讀取 log
    log_content = None
    if log_file.exists():
        log_content = log_file.read_bytes()

    # 7. 清理 session 目錄（在容器內刪除並 commit）
    # 必須在容器內主動刪除並 commit，否則容器 shutdown 時的 auto-commit 會把
    # output / log 等臨時檔案重新寫回 Volume，導致 session 目錄殘留。
    import shutil
    try:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        volume.commit()
    except Exception as _cleanup_err:
        print(f"[sakura_modal] 清理 session 目錄失敗 {session_dir}: {_cleanup_err}", flush=True)

    return {
        "translated_content": translated_content,
        "log_content": log_content,
        "elapsed_translate": elapsed_translate,
        "input_chars": input_chars,
        "output_chars": output_chars,
    }


if __name__ == "__main__":
    sys.exit(main())
