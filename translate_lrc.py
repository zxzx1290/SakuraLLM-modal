"""LRC 歌詞翻譯腳本

讀取 .lrc 歌詞檔，解析每行開頭的時間標籤（如 [00:12.34]、[01:02.500]，
可多個連用），只翻譯後面的歌詞文字，翻完後再把原始時間標籤接回去。
metadata 標籤（[ti:]、[ar:]、[al:] 等）、空行、無時間標籤的行皆原樣保留。
輸出僅含譯文的 LRC 檔（時間標籤不變、與原檔行數一致）。
"""
import re
import time
from argparse import ArgumentParser
from pathlib import Path

from dacite import from_dict
from transformers import GenerationConfig

import utils
import utils.cli
import utils.model as M

# 重用 MTool 腳本的批次翻譯、字典載入與待翻譯判斷邏輯（兩腳本同目錄）
from translate_mtool import load_gpt_dict, should_translate, translate_entries

# 行首一個或多個時間標籤，例如 [00:12.34] 或 [00:12.34][00:50.00]
# 時間標籤以數字開頭，可藉此與 metadata 標籤（[ti:]、[ar:] 等）區分。
TIME_TAG_RE = re.compile(r"^((?:\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\])+)(.*)$")


def parse_lrc(text: str) -> list[tuple[str | None, str]]:
    """將 LRC 文字拆成 (時間標籤前綴, 歌詞) 清單。

    前綴為 None 表示該行無時間標籤（metadata、空行等），整行原樣保留於第二欄。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parsed: list[tuple[str | None, str]] = []
    for line in text.split("\n"):
        m = TIME_TAG_RE.match(line)
        if m:
            parsed.append((m.group(1), m.group(2)))
        else:
            parsed.append((None, line))
    return parsed


def main():
    def extra_args(parser: ArgumentParser):
        g = parser.add_argument_group("LRC")
        g.add_argument("--data_path", type=str, default="data.lrc",
                       help="待翻譯的 LRC 檔案路徑。")
        g.add_argument("--output_path", type=str, default="data_translated.lrc",
                       help="輸出 LRC 檔案路徑。")
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

    raw = Path(args.data_path).read_text(encoding="utf-8", errors="replace")
    parsed = parse_lrc(raw)

    total_lyric_lines = sum(1 for prefix, _ in parsed if prefix is not None)
    # 去重後的待翻譯歌詞文字（保持出現順序）
    seen: set[str] = set()
    keys_to_translate: list[str] = []
    for prefix, lyric in parsed:
        if prefix is None:
            continue
        if lyric.strip() and should_translate(lyric) and lyric not in seen:
            seen.add(lyric)
            keys_to_translate.append(lyric)

    print(
        f"共 {len(parsed)} 行，含時間標籤歌詞 {total_lyric_lines} 行，"
        f"去重後待翻譯 {len(keys_to_translate)} 條"
    )

    translations: dict[str, str] = {}
    if keys_to_translate:
        t_start = time.time()
        translations = translate_entries(
            sakura_model, generation_config,
            keys_to_translate, args.text_length, gpt_dict,
        )
        elapsed = time.time() - t_start
        translated_chars = sum(len(v) for v in translations.values())
        speed = translated_chars / elapsed if elapsed > 0 else 0
        print(f"翻譯完成，耗時 {elapsed:.1f}s，輸出 {translated_chars} 字元，速度 {speed:.1f} 字元/秒")

    out_lines: list[str] = []
    for prefix, lyric in parsed:
        if prefix is None:
            out_lines.append(lyric)
        elif lyric in translations:
            out_lines.append(prefix + translations[lyric])
        else:
            # 空行、純標點、未翻譯（行數不符等）→ 保留原歌詞
            out_lines.append(prefix + lyric)

    Path(args.output_path).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"結果已寫入：{args.output_path}")


if __name__ == "__main__":
    main()
