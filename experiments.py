
"""
LLM Rule‑Articulation Experiments (Anthropic Claude)
===================================================

This standalone script defines **20 hand‑crafted text‑classification rules**,
creates a **train / test split** for each, and then:

1.  Feeds the **train split** to Claude as few‑shot in‑context examples using the
    tutorial‑recommended *user → assistant* message pairs.
2.  Evaluates Claude’s zero‑chain‑of‑thought predictions on the **test split**.
3.  Prompts Claude to articulate the underlying rule in a single English
    sentence.
4.  Prints accuracy and the articulated rule per task plus a Pandas summary.

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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import anthropic
import pandas as pd

###############################################################################
# Anthropic client (replace YOUR_API_KEY or set environment variable)
###############################################################################

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))
MODEL_ID = "claude-3-opus-20240229"  # change to opus/haiku if you wish

###############################################################################
# Task definition with explicit train / test splits
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
        """Return k/2 positive + k/2 negative examples from TRAIN sets."""
        assert k % 2 == 0, "k must be even"
        pos = random.sample(self.train_pos, k // 2)
        neg = random.sample(self.train_neg, k // 2)
        examples = [(x, "True") for x in pos] + [(x, "False") for x in neg]
        random.shuffle(examples)
        return examples

    def test_pairs(self) -> List[Tuple[str, str]]:
        """All held‑out examples with gold labels."""
        return [(x, "True") for x in self.test_pos] + [(x, "False") for x in self.test_neg]

###############################################################################
# Helper: split positives / negatives 50‑50 into train / test
###############################################################################

def split_examples(pos: List[str], neg: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    half_p = len(pos) // 2
    half_n = len(neg) // 2
    return pos[:half_p], neg[:half_n], pos[half_p:], neg[half_n:]

###############################################################################
# Hand‑crafted rules (20)
###############################################################################

def make_tasks() -> Dict[str, Task]:
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
    for name, (rule, pos, neg) in RAW.items():
        tr_pos, tr_neg, te_pos, te_neg = split_examples(pos, neg)
        tasks[name] = Task(name, rule, tr_pos, tr_neg, te_pos, te_neg)
    return tasks

###############################################################################
# Prompt builders (Anthropic tutorial style)
###############################################################################

SYSTEM_CLASSIFY = "You are a concise classifier. Respond with exactly 'True' or 'False'."
SYSTEM_EXPLAIN  = "You are a helpful analyst. In ONE short English sentence, describe the rule."

def build_classification_msgs(task: Task, query: str, k: int = 4):
    msgs: List[Dict[str, str]] = []
    for txt, lab in task.sample_few_shot(k):
        msgs.append({"role": "user", "content": f"Input: {txt}"})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": f"Input: {query}"})
    return msgs

def build_explanation_prompt(task: Task) -> str:
    lines: List[str] = [
        "Here are labelled examples. Describe the rule in ONE sentence.",
        "### Data",
    ]
    for txt in task.train_pos + task.test_pos:
        lines.append(f"Input: {txt}\\nLabel: True\\n")
    for txt in task.train_neg + task.test_neg:
        lines.append(f"Input: {txt}\\nLabel: False\\n")
    lines.append("### Question\\nWhat is the rule?")
    return "\\n".join(lines)

###############################################################################
# Claude wrappers
###############################################################################

def classify(task: Task, query: str, k: int = 4) -> str:
    resp = client.messages.create(
        model=MODEL_ID,
        system=SYSTEM_CLASSIFY,
        messages=build_classification_msgs(task, query, k),
        max_tokens=1,
    )
    return resp.content[0].text.strip()

def articulate_rule(task: Task) -> str:
    resp = client.messages.create(
        model=MODEL_ID,
        system=SYSTEM_EXPLAIN,
        messages=[{"role": "user", "content": build_explanation_prompt(task)}],
        max_tokens=100,
    )
    return resp.content[0].text.strip()

###############################################################################
# Experiment driver
###############################################################################

def run_experiments(k_shot: int = 4):
    random.seed(0)
    tasks = make_tasks()
    rows = []

    for task in tasks.values():
        tests = task.test_pairs()
        accuracy = sum(classify(task, txt, k_shot) == lbl for txt, lbl in tests) / len(tests)
        rule = articulate_rule(task)
        rows.append(dict(Task=task.name, Accuracy=accuracy, ArticulatedRule=rule))
        print(f"{task.name:25}  acc={accuracy:5.1%}  rule→ {rule}")

    print("\\n=== Summary ===")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    run_experiments()
