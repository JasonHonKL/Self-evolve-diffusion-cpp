#!/usr/bin/env python3
"""Generate golden token ids for the UMT5 sentencepiece tokenizer.

Golden = AutoTokenizer default call (what Wan2.2 does): Viterbi ids + </s>.
Cross-checks the rust fast tokenizer agree on every prompt (they do while no
prompt contains an out-of-vocab char; transformers 5.x slow glue has an
unk_id=2 bug for UMT5, see report).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_DIR = os.path.join(ROOT, "ckpts", "umt5_tokenizer")
OUT = os.path.join(ROOT, "tools", "golden", "tokenizer_golden.npz")

PROMPTS = [
    "A cat riding a skateboard through a neon city at night",
    "sunset over the mountains, cinematic film still, 35mm",
    "Hello, world! How's it going? 100% sure...",
    "3 dogs running; 42 birds flying & 7 fish swimming!!!",
    "antidisestablishmentarianism floccinaucinihilipilification pneumonoultramicroscopicsilicovolcanoconiosis",
    "日本語のテスト：東京タワーと富士山の風景",
    "中文测试：长城上的日出，电影质感",
    "한국어 테스트: 한강에서 자전거를 타는 사람들",
    "اختبار اللغة العربية: غروب الشمس في الصحراء",
    "हिन्दी परीक्षण: ताजमहल का सूर्योदय",
    "Русский тест: рассвет над Москвой-рекой",
    "naïve café résumé — façade, jalapeño",
    "emoji mix 🚀🌟🎉 fire and rain 🌧✨",
    "1234567890 + 0987654321 = numbers 3.14159",
    "2024-10-08T21:30:00Z timestamp, $1,234.56 EUR €",
    "A    lot    of     inner spaces after collapse",
    "MiXeD CaSe WoRdS and ALLCAPS WORDS",
    'code snippet: for(int i=0;i<10;i++){printf("%d\\n",i);}',
    "URL: https://example.com/path?query=1&other=2#frag",
    "le chat noir marche dans la rue à Paris",
    "Die schöne Straße in München — über allem",
    "Vì sao bầu trời màu xanh? Hỏi thế nào",
    "คนไทยทดสอบภาษาไทย",
    "multilingual mix: 猫と犬 chat et chien Hund и кот собака",
]


def main():
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    fast = Tokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))
    print(f"tokenizer: {type(tok).__name__}")

    ids, off = [], [0]
    for i, p in enumerate(PROMPTS):
        got = tok(p)["input_ids"]
        fast_ids = fast.encode(p).ids
        if got != fast_ids:
            print(f"PROMPT {i}: fast/slow MISMATCH\n slow={got}\n fast={fast_ids}")
            sys.exit(1)
        if 2 in got or 3 in got:  # unk / transformers-buggy unk: prompt has OOV char
            print(f"PROMPT {i} contains out-of-vocab char, ids={got}")
            sys.exit(1)
        ids += got
        off.append(len(ids))
        print(f"[{i:2d}] {len(got):3d} ids  {p[:60]!r}")

    blob = json.dumps(PROMPTS, ensure_ascii=False).encode()
    np.savez(
        OUT,
        prompts_json=np.frombuffer(blob, dtype=np.uint8),
        ids_flat=np.array(ids, dtype=np.int64),
        offsets=np.array(off, dtype=np.int64),
    )
    print(f"wrote {OUT}: {len(PROMPTS)} prompts, {len(ids)} total ids")


if __name__ == "__main__":
    main()
