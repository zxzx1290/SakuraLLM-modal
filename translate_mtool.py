"""MTool JSON 翻譯腳本

讀取 MTool 導出的 ManualTransFile.json（格式：{原文: 譯文}），
翻譯所有 value == key（MTool 格式未翻譯佔位）的條目，輸出包含譯文的 JSON 檔案。
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

from dacite import from_dict
from tqdm import tqdm
from transformers import GenerationConfig

import utils
import utils.cli
import utils.model as M
import utils.consts as consts


EXCLUDE_SUFFIXES = frozenset([
    '.mp3', '.wav', '.ogg', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
    '.json', '.txt', '.xml', '.html', '.css', '.js', '.zip', '.rar', '.dat',
])
EXCLUDE_PREFIXES = ('MapData/', 'SE/', 'BGS', 'BGM/', 'FIcon/', 'EV0', '\\img',
                    '<input type=', '<div ', 'width:')

_PURE_NEWLINES = {"\n", "\\n", "\r", "\\r", "\r\n", "\\r\\n"}

_PUNCTUATION = set(
    ' \u3000!"#$%&\'()*+,-./'
    '，。：；＜＝＞？＠'
    ':;<=>?@[\\]^_`{|}~—・？↑←↓→'
    '「」『』【】《》！＂＃＄％＆＇（）＊＋，－．／：；'
)


def _is_punctuation_only(s: str) -> bool:
    return bool(s) and all(c in _PUNCTUATION for c in s)


def should_translate(key: str) -> bool:
    """判斷此 MTool key 是否需要翻譯（過濾素材路徑、純數字、純標點等）"""
    if not key or key.strip() == "":
        return False
    if key.isdigit():
        return False
    if key.strip() in _PURE_NEWLINES:
        return False
    if _is_punctuation_only(key):
        return False
    suffix = ('.' + key.rstrip().rsplit('.', 1)[-1].lower()) if '.' in key else ''
    if suffix in EXCLUDE_SUFFIXES:
        return False
    if any(key.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    return True


def load_mtool_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MTool JSON 格式錯誤：期望最外層為 dict，得到 {type(data)}")
    bad = [(k, v) for k, v in data.items() if not isinstance(k, str) or not isinstance(v, str)]
    if bad:
        raise ValueError(f"MTool JSON 格式錯誤：有 {len(bad)} 個條目的 key 或 value 不是字串")
    return data


def load_gpt_dict(path: str) -> list:
    gpt_dict = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, temp = line.split("->", 1)
            if "#" in temp:
                dst, info = temp.split("#", 1)
                gpt_dict.append({"src": src.strip(), "dst": dst.strip(), "info": info.strip()})
            else:
                gpt_dict.append({"src": src.strip(), "dst": temp.strip()})
    return gpt_dict


def get_model_response(model, tokenizer, prompt, model_version, generation_config, text_length, llama_cpp):
    """與 translate_novel.py 相同的推理函式（含防退化重試）"""
    backup_configs = [
        GenerationConfig(
            temperature=0.1, top_p=0.3, top_k=40, num_beams=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            max_new_tokens=text_length, min_new_tokens=1, do_sample=True,
            repetition_penalty=1.0, frequency_penalty=0.05,
        ),
        GenerationConfig(
            temperature=0.1, top_p=0.3, top_k=40, num_beams=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            max_new_tokens=text_length, min_new_tokens=1, do_sample=True,
            repetition_penalty=1.0, frequency_penalty=0.2,
        ),
    ]

    if llama_cpp:
        def _gen(cfg):
            kwargs = dict(
                max_tokens=cfg.__dict__["max_new_tokens"],
                temperature=cfg.__dict__["temperature"],
                top_p=cfg.__dict__["top_p"],
                repeat_penalty=cfg.__dict__["repetition_penalty"],
            )
            if "frequency_penalty" in cfg.__dict__:
                kwargs["frequency_penalty"] = cfg.__dict__["frequency_penalty"]
            return model.model(prompt, **kwargs)

        output = _gen(generation_config)
        for cfg in backup_configs:
            if output["usage"]["completion_tokens"] < text_length:
                break
            print("model degeneration detected, retrying...")
            output = _gen(cfg)
        return output["choices"][0]["text"]

    generation = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        generation_config=generation_config,
    )[0]
    for cfg in backup_configs:
        if not utils.detect_degeneration(list(generation), model_version):
            break
        generation = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device),
            generation_config=cfg,
        )[0]
    response = tokenizer.decode(generation)
    return utils.split_response(response, model_version)


def build_batches(keys: list[str], text_length: int) -> list[list[str]]:
    """將待翻譯的 key 清單按 text_length 分批。
    含換行的條目單獨成一批，以確保能正確拆回。
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for key in keys:
        # 含換行的條目無法用 split("\n") 拆回，單獨批次
        if "\n" in key:
            if current:
                batches.append(current)
                current, current_len = [], 0
            batches.append([key])
            continue
        key_len = len(key)
        if current and current_len + key_len > text_length:
            batches.append(current)
            current, current_len = [key], key_len
        else:
            current.append(key)
            current_len += key_len
    if current:
        batches.append(current)
    return batches


def translate_single(
    sakura_model, generation_config: GenerationConfig,
    key: str, text_length: int, gpt_dict: list,
) -> str:
    prompt = consts.get_prompt(
        raw_jp_text=key,
        model_name=sakura_model.cfg.model_name,
        model_version=sakura_model.cfg.model_version,
        model_quant=sakura_model.cfg.model_quant,
        gpt_dict=gpt_dict,
    )
    return get_model_response(
        sakura_model.model, sakura_model.tokenizer,
        prompt, sakura_model.cfg.model_version,
        generation_config, text_length, sakura_model.cfg.llama_cpp,
    ).strip()


def translate_entries(
    sakura_model, generation_config: GenerationConfig,
    keys_to_translate: list[str], text_length: int, gpt_dict: list,
) -> dict[str, str]:
    """翻譯所有待翻譯的 key，回傳 {原文: 譯文} dict"""
    batches = build_batches(keys_to_translate, text_length)
    result: dict[str, str] = {}

    for batch in tqdm(batches, desc="翻譯進度"):
        if len(batch) == 1:
            result[batch[0]] = translate_single(sakura_model, generation_config, batch[0], text_length, gpt_dict)
            continue

        source_text = "\n".join(batch)
        prompt = consts.get_prompt(
            raw_jp_text=source_text,
            model_name=sakura_model.cfg.model_name,
            model_version=sakura_model.cfg.model_version,
            model_quant=sakura_model.cfg.model_quant,
            gpt_dict=gpt_dict,
        )
        output = get_model_response(
            sakura_model.model, sakura_model.tokenizer,
            prompt, sakura_model.cfg.model_version,
            generation_config, text_length, sakura_model.cfg.llama_cpp,
        ).strip()

        translated_lines = output.split("\n")
        if len(translated_lines) == len(batch):
            for orig, trans in zip(batch, translated_lines):
                result[orig] = trans.strip()
        else:
            # 行數不符，逐條重譯
            print(
                f"警告：翻譯行數不符（原 {len(batch)} 行，譯 {len(translated_lines)} 行），逐條重試..."
            )
            for orig in batch:
                result[orig] = translate_single(sakura_model, generation_config, orig, text_length, gpt_dict)

    return result


def main():
    def extra_args(parser: ArgumentParser):
        g = parser.add_argument_group("MTool")
        g.add_argument("--data_path", type=str, default="ManualTransFile.json",
                       help="MTool 導出的 JSON 檔案路徑。")
        g.add_argument("--output_path", type=str, default="ManualTransFile_translated.json",
                       help="輸出 JSON 檔案路徑。")
        g.add_argument("--text_length", type=int, default=512,
                       help="每次推理的最大文字長度（預設 512）。")
        g.add_argument("--gpt_dict_path", type=str, default=None,
                       help="術語表路徑（格式：原文->譯文 或 原文->譯文#備註）。")
        g.add_argument("--temperature", type=float, default=None,
                       help="採樣溫度（覆蓋模型預設值）。")
        g.add_argument("--top_p", type=float, default=None,
                       help="Top-p 採樣（覆蓋模型預設值）。")

    args = utils.cli.parse_args(do_validation=True, add_extra_args_fn=extra_args)

    import coloredlogs
    coloredlogs.install(level="INFO")

    cfg = from_dict(data_class=M.SakuraModelConfig, data=args.__dict__)
    sakura_model = M.SakuraModel(cfg=cfg)

    gpt_dict: list = []
    if args.gpt_dict_path:
        gpt_dict = load_gpt_dict(args.gpt_dict_path)
        print(f"loaded gpt dict: {len(gpt_dict)} entries")

    generation_config = GenerationConfig(
        temperature=args.temperature if args.temperature is not None else 0.1,
        top_p=args.top_p if args.top_p is not None else 0.3,
        top_k=40,
        num_beams=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=args.text_length,
        min_new_tokens=1,
        do_sample=True,
    )

    data = load_mtool_json(args.data_path)
    total = len(data)
    # MTool 格式：value == key（以原文作為未翻譯佔位）
    keys_to_translate = [k for k, v in data.items() if v == k and should_translate(k)]
    already_done = total - len(keys_to_translate)
    print(f"共 {total} 條，已翻譯 {already_done} 條，待翻譯 {len(keys_to_translate)} 條")

    if keys_to_translate:
        t_start = time.time()
        translations = translate_entries(
            sakura_model, generation_config,
            keys_to_translate, args.text_length, gpt_dict,
        )
        elapsed = time.time() - t_start
        for k, v in translations.items():
            data[k] = v
        translated_chars = sum(len(v) for v in translations.values())
        speed = translated_chars / elapsed if elapsed > 0 else 0
        print(f"翻譯完成，耗時 {elapsed:.1f}s，輸出 {translated_chars} 字元，速度 {speed:.1f} 字元/秒")

    Path(args.output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"結果已寫入：{args.output_path}")


if __name__ == "__main__":
    main()
