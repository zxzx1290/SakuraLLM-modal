from argparse import ArgumentParser
from dacite import from_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
from tqdm import tqdm

import utils
import utils.cli
import utils.model as M
import utils.consts as consts


total_token = 0
generation_time = 0
def add_token_cnt(cnt):
    global total_token
    total_token += cnt

def add_time(time):
    global generation_time
    generation_time += time

def get_novel_text_list(data_path, text_length):
    data_list = list()
    with open(data_path, 'r', encoding="utf-8") as f:
        data = f.read()
    data = data.strip()
    # Normalize 3+ consecutive newlines to 2 (at most one blank line between paragraphs)
    data_raw = re.sub('\n{3,}', '\n\n', data)
    # Build translation chunks from content lines only (blank lines are pass-through)
    content_lines = [l for l in data_raw.split("\n") if l.strip()]
    print(f"text total words: {len(''.join(content_lines))}")
    i = 0
    while i < len(content_lines):
        r = text_length
        text = ""
        while len(text) < r:
            if i >= len(content_lines):
                break
            if len(text) > max(- len(content_lines[i]) + r, 0):
                break
            else:
                text += content_lines[i] + "\n"
                i += 1
        text = text.strip()
        data_list.append(text)
    return data_raw, data_list

def get_model_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, llama_cpp: bool):
    backup_generation_config_stage2 = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=text_length,
            min_new_tokens=1,
            do_sample=True,
            repetition_penalty=1.0,
            frequency_penalty=0.05
        )

    backup_generation_config_stage3 = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=text_length,
            min_new_tokens=1,
            do_sample=True,
            repetition_penalty=1.0,
            frequency_penalty=0.2
        )


    backup_generation_config = [backup_generation_config_stage2, backup_generation_config_stage3]

    if llama_cpp:

        def generate(model, generation_config):
            if "frequency_penalty" in generation_config.__dict__.keys():
                output = model.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'], frequency_penalty=generation_config.__dict__['frequency_penalty'])
            else:
                output = model.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'])
            return output

        stage = 0
        output = generate(model, generation_config)
        while output['usage']['completion_tokens'] == text_length:
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            print("model degeneration detected, retrying...")
            output = generate(model, backup_generation_config[stage-1])
        response = output['choices'][0]['text']
        return response

    generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=generation_config)[0]
    if len(generation) > text_length:
        stage = 0
        while utils.detect_degeneration(list(generation), model_version):
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=backup_generation_config[stage-1])[0]
    response = tokenizer.decode(generation)
    output = utils.split_response(response, model_version)

    return output

def restore_blank_lines(original_text: str, translated_text: str) -> str:
    """Reinsert blank lines into translated output using the original text structure."""
    original_lines = original_text.split("\n")
    translated_content = [l for l in translated_text.strip().split("\n") if l.strip()]
    result = []
    trans_idx = 0
    for orig_line in original_lines:
        if orig_line.strip():
            result.append(translated_content[trans_idx] if trans_idx < len(translated_content) else "")
            trans_idx += 1
        else:
            result.append("")
    return "\n".join(result).strip()

def get_compare_text(source_text, translated_text):
    source_lines = source_text.split("\n")
    source_content = [l for l in source_lines if l.strip()]
    translated_content = [l for l in translated_text.strip().split("\n") if l.strip()]
    if len(source_content) != len(translated_content):
        print(f"error occurred when output compared text(length of source is {len(source_content)} while length of translated is {len(translated_content)}), fallback to output only translated text.")
        return translated_text
    output_text = ""
    trans_idx = 0
    for src_line in source_lines:
        if src_line.strip():
            output_text += src_line + "\n" + translated_content[trans_idx] + "\n\n"
            trans_idx += 1
        else:
            output_text += "\n"
    return output_text.strip()


def load_gpt_dict(path: str) -> list:
    """載入字典檔（每行：原文->譯文 或 原文->譯文#備註，# 開頭為註解行）"""
    gpt_dict = []
    with open(path, 'r', encoding='utf-8') as f:
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


def main():
    def extra_args(parser: ArgumentParser):
        novel_group = parser.add_argument_group("Novel")
        novel_group.add_argument("--data_path", type=str, default="data.txt", help="file path of the text you want to translate.")
        novel_group.add_argument("--output_path", type=str, default="data_translated.txt", help="save path of the text model translated.")
        novel_group.add_argument("--compare_text", action="store_true", help="whether to output with both source text and translated text in order to compare.")
        novel_group.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")
        novel_group.add_argument("--gpt_dict_path", type=str, default=None, help="path to glossary dict file (format: src->dst or src->dst#info).")
        novel_group.add_argument("--temperature", type=float, default=None, help="sampling temperature (overrides model default).")
        novel_group.add_argument("--top_p", type=float, default=None, help="top-p sampling (overrides model default).")

    args = utils.cli.parse_args(do_validation=True, add_extra_args_fn=extra_args)

    import coloredlogs
    coloredlogs.install(level="INFO")

    cfg = from_dict(data_class=M.SakuraModelConfig, data=args.__dict__)
    sakura_model = M.SakuraModel(cfg=cfg)

    gpt_dict = []
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
        max_new_tokens=512,
        min_new_tokens=1,
        do_sample=True
    )

    print("translating...")
    with open(args.output_path, 'w', encoding='utf-8') as f_w:
        start = time.time()

        data_raw, data_list = get_novel_text_list(args.data_path, args.text_length)
        data = ""
        for d in tqdm(data_list):
            prompt = consts.get_prompt(
                raw_jp_text=d,
                model_name=sakura_model.cfg.model_name,
                model_version=sakura_model.cfg.model_version,
                model_quant=sakura_model.cfg.model_quant,
                gpt_dict=gpt_dict,
            )
            #FIXME(kuriko): refactor this to sakura_model.completion()
            output = get_model_response(
                sakura_model.model,
                sakura_model.tokenizer,
                prompt,
                sakura_model.cfg.model_version,
                generation_config,
                sakura_model.cfg.text_length,
                sakura_model.cfg.llama_cpp,
            )
            data += output.strip() + "\n"

        end = time.time()
        print("translation completed, used time: ", generation_time, end-start, ", total tokens: ", total_token, ", speed: ", total_token/(end-start), " token/s")

        print("saving...")
        if args.compare_text:
            f_w.write(get_compare_text(data_raw, data))
        else:
            f_w.write(restore_blank_lines(data_raw, data))

    print("completed.")

if __name__ == "__main__":

    main()
