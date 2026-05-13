"""Fix unescaped inner double-quotes in segment JSON files produced by extraction agents.

Strategy: scan char-by-char tracking whether we're inside a string value. When inside
and we hit a `"` that isn't followed by JSON-structural chars (`,`, `}`, `]`, `:`, whitespace+structural),
treat it as a literal quote in the Chinese text and replace with U+201C / U+201D pair.
"""
import json
import sys
from pathlib import Path

def fix(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_string = False
    inner_quote_count = 0  # how many inner quotes seen in current string
    while i < n:
        c = text[i]
        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
                inner_quote_count = 0
            i += 1
            continue
        # in_string == True
        if c == '\\':
            # escape sequence: keep next char literal
            out.append(c)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == '"':
            # Lookahead: structural follower means string ended; otherwise it's a stray inner quote.
            j = i + 1
            while j < n and text[j] in ' \t\r\n':
                j += 1
            follower = text[j] if j < n else ''
            if follower in ',:}]' or j == n:
                out.append('"')
                in_string = False
                inner_quote_count = 0
                i += 1
            else:
                # stray quote inside a value — replace with curly quote (alternating open/close)
                out.append('“' if inner_quote_count % 2 == 0 else '”')
                inner_quote_count += 1
                i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)

def main():
    seg_dir = Path(r"D:\BaiduSyncdisk\CUHKSZ\Y4T2\DDA4080\GraphRAG\data\memoirs\segments")
    for fp in sorted(seg_dir.glob("M*.json")):
        original = fp.read_text(encoding='utf-8')
        try:
            json.loads(original)
            print(f"OK   {fp.name} (no fix needed)")
            continue
        except json.JSONDecodeError as e:
            pass
        fixed = fix(original)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"FAIL {fp.name}: still invalid after fix — {e}")
            # save fixed attempt to .broken for inspection
            (fp.with_suffix('.broken.json')).write_text(fixed, encoding='utf-8')
            continue
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"FIX  {fp.name}: {len(data)} segments")

if __name__ == "__main__":
    main()
