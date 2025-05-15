"""
LLM Rule‑Articulation Experiments (Anthropic Claude)
===================================================

This script now auto‑generates **100 simple, hand‑crafted classification rules**
using parameterised templates (letters, digits, punctuation, length parity,
start/ends‑with, etc.). Each rule has a 6‑example train split (3 pos / 3 neg)
and a 6‑example test split (3 pos / 3 neg).

You can scale `NUM_RULES` to any value ≤ 100 without hand‑editing.

Run
---
```bash
pip install anthropic pandas
export ANTHROPIC_API_KEY="sk‑…"
python rule_articulation.py
```
"""

from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

import anthropic
import pandas as pd
from tqdm import tqdm

###############################################################################
# Config
###############################################################################

NUM_RULES = 100              # generate up to 100 rules
POS_NEG_PER_SPLIT = 3        # 3 pos / 3 neg in train and test each
MODEL_ID = "claude-3-opus-20240229"

###############################################################################
# Anthropic client
###############################################################################

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))

###############################################################################
# Dataclass
###############################################################################

@dataclass
class Task:
    name: str
    rule: str
    train_pos: List[str]
    train_neg: List[str]
    test_pos: List[str]
    test_neg: List[str]

    def sample_few_shot(self, k: int = 4) -> List[Tuple[str, str]]:
        assert k % 2 == 0
        pos = random.sample(self.train_pos, k // 2)
        neg = random.sample(self.train_neg, k // 2)
        pairs = [(x, "True") for x in pos] + [(x, "False") for x in neg]
        random.shuffle(pairs)
        return pairs

    def test_pairs(self):
        return [(x, "True") for x in self.test_pos] + [(x, "False") for x in self.test_neg]

###############################################################################
# Utility generators for synthetic examples
###############################################################################

def gen_string_without(chars: str, length: int = 6):
    pool = [c for c in string.ascii_lowercase if c not in chars]
    return "".join(random.choice(pool) for _ in range(length))

def contains_letter_rule(letter: str, idx: int) -> Task:
    rule = f"Label True iff the input contains the letter '{letter}'."
    pos_examples = [f"{letter}{gen_string_without(letter,4)}", f"{gen_string_without(letter,4)}{letter}", f"mix{letter}mix", f"hello {letter}"]
    neg_examples = [gen_string_without(letter,5) for _ in range(4)]
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)
    tr_p, te_p = pos_examples[:POS_NEG_PER_SPLIT], pos_examples[POS_NEG_PER_SPLIT:]
    tr_n, te_n = neg_examples[:POS_NEG_PER_SPLIT], neg_examples[POS_NEG_PER_SPLIT:]
    return Task(f"contains_{letter}_{idx}", rule, tr_p, tr_n, te_p, te_n)

def starts_with_letter_rule(letter: str, idx: int) -> Task:
    rule = f"Label True iff the input starts with '{letter}'."
    pos = [f"{letter}{gen_string_without('',4)}" for _ in range(4)]
    neg = [f"{random.choice([c for c in string.ascii_lowercase if c!=letter])}{gen_string_without('',4)}" for _ in range(4)]
    tr_p, te_p = pos[:POS_NEG_PER_SPLIT], pos[POS_NEG_PER_SPLIT:]
    tr_n, te_n = neg[:POS_NEG_PER_SPLIT], neg[POS_NEG_PER_SPLIT:]
    return Task(f"starts_{letter}_{idx}", rule, tr_p, tr_n, te_p, te_n)

def even_length_rule(idx: int) -> Task:
    rule = "Label True iff the character count is even."
    evens = ["aa", "bbbb", "cccccc", "dddddddd"]
    odds  = ["a", "bbb", "ccccc", "ddddddd"]
    tr_p, te_p = evens[:POS_NEG_PER_SPLIT], evens[POS_NEG_PER_SPLIT:]
    tr_n, te_n = odds[:POS_NEG_PER_SPLIT], odds[POS_NEG_PER_SPLIT:]
    return Task(f"even_len_{idx}", rule, tr_p, tr_n, te_p, te_n)

RULE_BUILDERS = [contains_letter_rule, starts_with_letter_rule, even_length_rule]

###############################################################################
# Build tasks
###############################################################################

def make_tasks() -> Dict[str, Task]:
    """Build tasks with 6 handcrafted train examples and 50 auto‑generated test
    examples (50 pos + 50 neg) using `example_factory.get_examples`."""
    from example_factory import get_examples  # import locally to avoid circular deps

    RAW: Dict[str, Tuple[str, List[str], List[str]]] = {
        "all_lowercase": (
            "Label is True *iff* the entire input is lowercase.",
            ["all lowercase", "hello world", "good morning", "machine learning", "data science", "openai rocks"],
            ["Some Upper", "MIXED case", "UPPER", "Hello", "PyTorch", "OpenAI"],
        ),
        "all_uppercase": (
            "Label is True *iff* the entire input is uppercase.",
            ["ALL UPPER", "HELLO WORLD", "GOOD MORNING", "MACHINE LEARNING", "DATA", "OPENAI"],
            ["some lower", "Mixed CASE", "lowercase", "Hello World", "TensorFlow", "Science"],
        ),
        "contains_number": (
            "Label is True *iff* the input contains at least one digit (0‑9).",
            ["room 101", "call me at 555", "version 2.0", "there are 3 cats", "2025 is soon", "lucky 7"],
            ["no digits here", "just text", "empty string", "numbers spelled out", "digitless", "hello"],
        ),
        "contains_color_word": (
            "Label is True *iff* the input contains an English color word.",
            ["blue sky", "red rose", "green grass", "bright yellow", "black cat", "white snow"],
            ["clear sky", "pinkish hue", "gray area", "sunny day", "colorless", "shade"],
        ),
        "starts_with_vowel": (
            "Label is True *iff* the first non‑space character is a vowel.",
            ["apple pie", "orange juice", "elephant", "umbrella stand", "igloo home", "octopus swim"],
            ["banana split", "carrot cake", "zebra", "lion den", "tiger run", "monkey bars"],
        ),
        "ends_with_exclamation": (
            "Label is True *iff* the input ends with an exclamation mark.",
            ["watch out!", "amazing!", "great job!", "done!", "wow!", "hello!"],
            ["watch out", "amazing", "great job", "done.", "wow", "hello"],
        ),
        "palindrome": (
            "Label is True *iff* the input (ignoring spaces/case) is a palindrome.",
            ["madam", "racecar", "level", "civic", "rotor", "noon"],
            ["palindrome", "python", "algorithm", "hello", "world", "openai"],
        ),
        "even_length": (
            "Label is True *iff* the character count is even.",
            ["abcd", "1234", "even", "length", "python3", "!!"],
            ["abc", "123", "odd", "hey", "!", "seven7"],
        ),
        "odd_length": (
            "Label is True *iff* the character count is odd.",
            ["abc", "123", "odd", "hello!", "five5", "seven7"],
            ["abcd", "1234", "even", "lengths", "python", "!!"],
        ),
        "contains_question": (
            "Label is True *iff* the input contains a question mark.",
            ["are you ok?", "what time?", "how?", "really?", "why not?", "ready?"],
            ["are you ok", "what time", "how", "really", "why not", "ready"],
        ),
        "contains_currency_symbol": (
            "Label is True *iff* the input contains $, €, £ or ¥.",
            ["cost is $5", "price €10", "worth £30", "amount ¥500", "$100 deal", "€2"],
            ["cost is five", "price ten", "worth thirty", "amount 500", "deal", "money"],
        ),
        "contains_emotion_word": (
            "Label is True *iff* the input contains an emotion word (happy, sad, angry).",
            ["I am happy", "she feels sad", "angry birds", "happy day", "so sad", "not angry"],
            ["emotionless", "neutral mood", "joyful", "upset", "content", "excited"],
        ),
        "multiple_of_three": (
            "Label is True *iff* the standalone number is a multiple of 3.",
            ["3", "6", "9", "12", "15", "18"],
            ["2", "4", "5", "7", "10", "11"],
        ),
        "prime_number": (
            "Label is True *iff* the standalone number is prime.",
            ["2", "3", "5", "7", "11", "13"],
            ["4", "6", "8", "9", "10", "12"],
        ),
        "ends_with_period": (
            "Label is True *iff* the input ends with a period.",
            ["end.", "statement.", "done.", "finish.", "stop.", "close."],
            ["end", "statement", "done!", "finish", "stop", "close"],
        ),
        "contains_email": (
            "Label is True *iff* the input contains an '@' symbol (email).",
            ["email me@site.com", "contact a@b.com", "user@example.org", "hello@test.io", "name@mail.com", "x@y.z"],
            ["email me at site", "contact mail", "user example", "hello test", "name mail", "xyz"],
        ),
        "more_than_three_words": (
            "Label is True *iff* the input has more than three words.",
            ["this has four words", "five words are here", "count these six words", "seven words in this example", "exactly four word phrase", "lots of words in this sentence"],
            ["just three words", "only two", "one", "hello world", "hi there", "tiny phrase"],
        ),
        "contains_date": (
            "Label is True *iff* the input contains a date like DD/MM or YYYY-MM-DD.",
            ["today is 01/01", "event on 2025-05-14", "birthday 19/05", "deadline 2024-12-31", "report 30/06", "1999-07-04 party"],
            ["today is tomorrow", "event soon", "birthday next week", "deadline year end", "report later", "party time"],
        ),
        "contains_hashtag": (
            "Label is True *iff* the input contains a hashtag (#).",
            ["love this #photo", "#sunset vibes", "check #AI", "using #python", "#fun times", "coding in #ML"],
            ["love this photo", "sunset vibes", "check AI", "using python", "fun times", "coding in ML"],
        ),
        "contains_url": (
            "Label is True *iff* the input contains http or https URL.",
            ["go to http://example.com", "visit https://openai.com", "check http://site.org", "https://api.ai", "link http://abc.xyz", "see https://test.com"],
            ["go to example", "visit openai", "check site", "api ai", "abc xyz", "link test"],
        ),
    }

    tasks: Dict[str, Task] = {}
    for name, (rule_text, pos_train, neg_train) in tqdm(RAW.items()):
        if os.path.exists(f"tasks/{name}.json"):
            print(f"Skipping {name} (already exists)")
            
        else:
            # Auto‑generate 50 positives & 50 negatives for TEST using example_factory
            pos_test, neg_test = get_examples(name, rule_text, 50)
            print(pos_test, neg_test)
            tasks[name] = Task(name, rule_text, pos_train, neg_train, pos_test, neg_test)
            save_single_task(tasks[name])  # save immediately
    return tasks

###############################################################################
# Prompt helpers
###############################################################################

SYSTEM_CLASSIFY = "You are a concise classifier. Respond with exactly 'True' or 'False'."
SYSTEM_EXPLAIN  = "You are a helpful analyst. In ONE short English sentence, describe the rule."

def build_classification_msgs(task: Task, query: str, k: int = 4):
    msgs = []
    for txt, lab in task.sample_few_shot(k):
        msgs.append({"role": "user", "content": f"Input: {txt}"})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": f"Input: {query}"})
    return msgs

def build_explanation_prompt(task: Task) -> str:
    lines = ["Here are labelled examples. Describe the rule in ONE sentence.", "### Data"]
    for txt in task.train_pos + task.train_neg:
        lab = "True" if txt in task.train_pos else "False"
        lines.append(f"Input: {txt}\\nLabel: {lab}\\n")
    lines.append("### Question\\nWhat is the rule?")
    return "\\n".join(lines)

###############################################################################
# Claude wrappers
###############################################################################

def classify(task: Task, query: str, k: int = 4) -> str:
    resp = client.messages.create(model=MODEL_ID, system=SYSTEM_CLASSIFY, messages=build_classification_msgs(task, query, k), max_tokens=1)
    return resp.content[0].text.strip()

def articulate_rule(task: Task) -> str:
    resp = client.messages.create(model=MODEL_ID, system=SYSTEM_EXPLAIN, messages=[{"role": "user", "content": build_explanation_prompt(task)}], max_tokens=100)
    return resp.content[0].text.strip()

def save_tasks(tasks: Dict[str, Task], path: str = "tasks.json") -> None:
    """Persist tasks dict to JSON for later reuse."""
    import json
    ser = {
        name: {
            "rule": t.rule,
            "train_pos": t.train_pos,
            "train_neg": t.train_neg,
            "test_pos": t.test_pos,
            "test_neg": t.test_neg,
        }
        for name, t in tasks.items()
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ser, fh, indent=2)
    print(f"Saved {len(tasks)} tasks → {path}")

def load_tasks(dir_path: str = "tasks") -> Dict[str, Task]:
    """Read all task JSON files and recreate Task objects."""
    import os, json
    tasks: Dict[str, Task] = {}
    for fname in os.listdir(dir_path):
        if fname.endswith(".json"):
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if len(data['test_pos']) == 0 or len(data['test_neg']) == 0:
                print(f"Skipping {fname} (no test examples)")
                continue
            else:
                print(f"Loading {fname}")
                tasks[fname[:-5]] = Task(
                    fname[:-5],
                    data["rule"],
                    data["train_pos"],
                    data["train_neg"],
                    data["test_pos"],
                    data["test_neg"],
                )
    return tasks

def save_single_task(task: Task, dir_path: str = "tasks") -> None:
    """Write one Task instance to <dir_path>/<task_name>.json."""
    import os, json, re
    os.makedirs(dir_path, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", task.name)
    out_path = os.path.join(dir_path, f"{safe}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "rule": task.rule,
            "train_pos": task.train_pos,
            "train_neg": task.train_neg,
            "test_pos": task.test_pos,
            "test_neg": task.test_neg,
        }, fh, indent=2)
    print(f"Saved task {task.name} → {out_path}")

###############################################################################
# Experiment driver
###############################################################################

def run_experiments(k_shot: int = 12):
    random.seed(0)
    tasks = load_tasks()
    rows = []
    for task in tasks.values():
        acc = sum(classify(task, txt, k_shot) == lab for txt, lab in task.test_pairs()) / len(task.test_pairs())
        rule = articulate_rule(task)
        rows.append(dict(Task=task.name, Accuracy=acc, Rule=rule))
        print(f"{task.name:20}  acc={acc:5.1%}  {rule}")
    print("\n=== Summary ===")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    run_experiments()
