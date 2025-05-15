import random, string, re, os

random.seed(0)  # reproducible
from tqdm import tqdm

VOWELS = "aeiou"
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
EMOTIONS = ["happy", "sad", "angry"]
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]
DOW = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

import anthropic, json, re

PROMPT_TEMPLATE = """
Below is a natural-language classification rule.

RULE:
"{RULE}"

Please produce **exactly {N}** positive examples (label=True) and **exactly {N}**
negative examples (label=False).  Requirements:

• Each example must be a single line of text ≤ 80 characters.
• The examples must be diverse – avoid trivial permutations.
• Return JSON with two top-level keys: "positive" and "negative",
  each mapping to a list of {N} strings.
• Do NOT include any extra keys or commentary.

JSON:
"""

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))

def llm_generate_examples(rule_text: str, n=50):
    prompt = (PROMPT_TEMPLATE
          .replace("{RULE}", rule_text)
          .replace("{N}", str(n)))
    try:
        resp = client.messages.create(
            model="claude-3-opus-20240229",
            system="You are a data-generation assistant.",
            max_tokens=4_000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"Error: {e}")
        return [], []
    try:
        data = json.loads(resp.content[0].text)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Response: {resp.content[0].text}")
        return [], []
    return data["positive"], data["negative"]

def word(n=5):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))

def number():
    return str(random.randint(0, 9999))

def join(*parts):
    return ' '.join(parts)

# ───────────────────────── predicate helpers ──────────────────────────
def is_palindrome(s: str):
    t = re.sub(r'[^a-z0-9]', '', s.lower())
    return t == t[::-1]

def is_prime(n: int):
    if n < 2: return False
    for p in range(2, int(n**0.5)+1):
        if n % p == 0: return False
    return True

# ───────────────────────── rule predicates ────────────────────────────
def rule_true(rule, s):
    s_lc = s.lower()
    if rule == "all_lowercase":              return s.islower()
    if rule == "all_uppercase":              return s.isupper()
    if rule == "starts_with_vowel":          return s_lc[0] in VOWELS
    if rule == "ends_with_exclamation":      return s.endswith("!")
    if rule == "ends_with_period":           return s.endswith(".")
    if rule == "contains_number":            return any(ch.isdigit() for ch in s)
    if rule == "contains_color_word":        return any(c in s_lc for c in COLORS)
    if rule == "contains_question":          return "?" in s
    if rule == "contains_currency_symbol":   return any(sym in s for sym in "$€£¥")
    if rule == "contains_emotion_word":      return any(e in s_lc for e in EMOTIONS)
    if rule == "multiple_of_three":          return s.isdigit() and int(s)%3==0
    if rule == "prime_number":               return s.isdigit() and is_prime(int(s))
    if rule == "palindrome":                 return is_palindrome(s)
    if rule == "even_length":                return len(s)%2==0
    if rule == "odd_length":                 return len(s)%2==1
    if rule == "contains_email":             return bool(re.search(r"\\w+@\\w+\\.\\w+", s))
    if rule == "more_than_three_words":      return len(s.split()) > 3
    if rule == "contains_date":              return bool(re.search(r"(\\d{2}/\\d{2})|\\d{4}-\\d{2}-\\d{2}", s))
    if rule == "contains_hashtag":           return "#" in s
    if rule == "contains_url":               return "http://" in s or "https://" in s
    if rule == "contains_color_word":        return any(c in s_lc for c in COLORS)
    # fallback
    return False

def verified_examples(rule_name, rule_text, n=50):
    pos, neg = llm_generate_examples(rule_text, n)
    if len(pos) == 0 and len(neg) == 0:
        print("No examples generated.")
        return [], []
    pos_ok = [s for s in pos if rule_true(rule_name, s)]
    neg_ok = [s for s in neg if not rule_true(rule_name, s)]
    print(f"pos: {len(pos_ok)} / {n} neg: {len(neg_ok)} / {n}")

    # ask again if too many failures
    attempts = 0
    while (len(pos_ok) < n // 2 or len(neg_ok) < n // 2) and attempts < 3:
        extra_pos, extra_neg = llm_generate_examples(rule_text, n)
        pos_ok += [s for s in extra_pos if rule_true(rule_name, s)]
        neg_ok += [s for s in extra_neg if not rule_true(rule_name, s)]
        attempts += 1

    return pos_ok[:n], neg_ok[:n]

def get_examples(rule_name, rule_text, n=50):
    pos,neg = verified_examples(rule_name, rule_text, n)
    return pos, neg
