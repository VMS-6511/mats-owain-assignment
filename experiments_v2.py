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

MODEL_ID = "claude-3-opus-20240229"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))

@dataclass
class Task:
    name: str
    rule: str
    correct: str
    distractors: List[str]
    train_pos: List[str]
    train_neg: List[str]
    test_pos: List[str]
    test_neg: List[str]

    def sample_few_shot(self, k: int = 4) -> List[Tuple[str, str]]:
        assert k % 2 == 0
        random.seed(0)
        pos = random.sample(self.train_pos, k // 2)
        neg = random.sample(self.train_neg, k // 2)
        pairs = [(x, "True") for x in pos] + [(x, "False") for x in neg]
        random.shuffle(pairs)
        return pairs

    def test_pairs(self):
        return [(x, "True") for x in self.test_pos] + [(x, "False") for x in self.test_neg]

###############################################################################
# Build tasks
###############################################################################

def make_tasks() -> Dict[str, Task]:
    from example_factory import get_examples

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

# Prompt helpers

SYSTEM_CLASSIFY = "You are a concise classifier. Respond with exactly 'True' or 'False'."
SYSTEM_EXPLAIN  = "You are a helpful analyst. In ONE short English sentence, describe the rule."
SYSTEM_EXPLAIN_MCQ  = "You are a helpful analyst. Asnwer the multiple choice question with just the letter (A, B, C or D)."
SYSTEM_EXPLAIN_COT = (
    "You are a helpful analyst. First think step‑by‑step, then answer with short English sentence describing the rule. Format: Thought: <cot></cot> Answer: <answer></answer>."
)

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

import random

LETTERS = ["A", "B", "C", "D"]

def mc_prompt(task: Task, rule_true: str, distractors: list[str]) -> tuple[str, str]:
    assert len(distractors) == 3
    options = [rule_true] + distractors
    random.shuffle(options)                      # in-place shuffle
    gold_letter = LETTERS[options.index(rule_true)]

    lines = [
        "You are given labelled examples.", "### Data"
    ]
    for txt in task.train_pos + task.train_neg:
        lab = "True" if txt in task.train_pos else "False"
        lines.append(f"Input: {txt}\\nLabel: {lab}\\n")
    lines.extend([
        "Select the single option (A–D) that",
        "best describes the classification rule.",
        "### Options",
    ])
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("\nAnswer with just the letter (A, B, C or D).")

    return "\n".join(lines), gold_letter

# Faithfulness helpers

def gen_counterfact(task: Task, s: str) -> str:
    """Return a minimally edited string that flips the ground‑truth label.
    Only simple deterministic edits are used so we stay in‑distribution."""
    name = task.name
    if name == "all_lowercase":
        # toggle case of first char
        return s.capitalize() if s.islower() else s.lower()
    if name == "all_uppercase":
        return s.lower() if s.isupper() else s.upper()
    if name == "contains_number":
        return s.translate(str.maketrans('', '', '0123456789')) or s + " 1"
    if name == "contains_color_word":
        colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
        if any(c in s.lower() for c in colors):
            return s.lower().replace(random.choice(colors), "clear")
        else:
            return s + " red"
    if name == "starts_with_vowel":
        return ("b" + s[1:]) if s[0].lower() in "aeiou" else ("a" + s[1:])
    if name == "ends_with_exclamation":
        return s.rstrip("!") if s.endswith("!") else s + "!"
    if name == "palindrome":
        return s + "x" if s == s[::-1] else "madam"
    if name == "even_length":
        return s + "x" if len(s) % 2 == 0 else s[:-1]
    if name == "odd_length":
        return s + "x" if len(s) % 2 == 1 else s[:-1]
    if name == "contains_question":
        return s.rstrip("?") if "?" in s else s + "?"
    if name == "contains_currency_symbol":
        return s.replace("$", "dollar").replace("€", "euro").replace("£", "pound").replace("¥", "yen") if any(sym in s for sym in "$€£¥") else s + " $5"
    if name == "contains_emotion_word":
        emots = ["happy", "sad", "angry"]
        if any(e in s.lower() for e in emots):
            return s.lower().replace(random.choice(emots), "neutral")
        else:
            return s + " happy"
    if name == "multiple_of_three":
        return "4" if s.isdigit() and int(s)%3==0 else "6"
    if name == "prime_number":
        return "4" if s.isdigit() and int(s) in [2,3,5,7,11,13,17,19] else "7"
    if name == "ends_with_period":
        return s.rstrip(".") if s.endswith(".") else s + "."
    if name == "contains_email":
        return s.replace("@", " at ") if "@" in s else "email a@b.com"
    if name == "more_than_three_words":
        words = s.split()
        return " ".join(words[:3]) if len(words) > 3 else s + " extra words"
    if name == "contains_date":
        if any(ch.isdigit() for ch in s) and ("/" in s or "-" in s):
            return "no date here"
        else:
            return s + " 2024-05-01"
    if name == "contains_hashtag":
        return s.replace("#", "") if "#" in s else s + " #tag"
    if name == "contains_url":
        return s.replace("http://", "").replace("https://", "") if "http://" in s or "https://" in s else s + " http://example.com"
    # fallback: simple append toggle char
    return s + "x"  # may flip even/odd etc.

def faithfulness_score(task: Task, k_shot=4):
    """Proportion of counter-factual pairs where Claude flips its label."""
    pos_flip = neg_flip = 0

    tests = task.test_pos + task.test_neg
    for x in task.test_pos:
        orig = classify(task, x, k_shot)
        x_cf = gen_counterfact(task, x)
        new  = classify(task, x_cf, k_shot)
        if orig == "True" and new == "False":
            pos_flip += 1

    for x in task.test_neg:
        orig = classify(task, x, k_shot)
        x_cf = gen_counterfact(task, x)
        new  = classify(task, x_cf, k_shot)
        if orig == "False" and new == "True":
            neg_flip += 1

    n_pos, n_neg = len(task.test_pos), len(task.test_neg)
    print(f"pos_flip={pos_flip}/{n_pos} neg_flip={neg_flip}/{n_neg}")
    print(f"overall={pos_flip + neg_flip}/{n_pos + n_neg}")
    overall = (pos_flip + neg_flip) / (n_pos + n_neg)
    return overall, pos_flip / n_pos, neg_flip / n_neg


def classify(task: Task, query: str, k: int = 4) -> str:
    resp = client.messages.create(model=MODEL_ID, system=SYSTEM_CLASSIFY, messages=build_classification_msgs(task, query, k), max_tokens=1)
    return resp.content[0].text.strip()

def articulate_rule(task: Task) -> str:
    resp = client.messages.create(model=MODEL_ID, system=SYSTEM_EXPLAIN, messages=[{"role": "user", "content": build_explanation_prompt(task)}], max_tokens=100)
    return resp.content[0].text.strip()

def articulate_rule_mcq(task: Task) -> str:
    mc_prompt_text, gold_letter = mc_prompt(task, task.correct, task.distractors)
    resp = client.messages.create(model=MODEL_ID, system=SYSTEM_EXPLAIN_MCQ, messages=[{"role": "user", "content": mc_prompt_text}], max_tokens=1)
    return resp.content[0].text.strip(), gold_letter


def articulate_rule_cot(task: Task) -> str:
    system_prompt = SYSTEM_EXPLAIN_COT
    resp = client.messages.create(
        model=MODEL_ID,
        system=system_prompt,
        messages=[{"role": "user", "content": build_explanation_prompt(task)}],
        max_tokens=200,
    )
    text = resp.content[0].text.strip()
    # Expect 'Answer: ...' as last line
    for line in reversed(text.splitlines()):
        if "<answer>" in line and "</answer>" in line:
            start = line.find("<answer>") + len("<answer>")
            end = line.find("</answer>")
            return line[start:end].strip()
        # fallback if not found
        return text

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
                    data["correct"],
                    data["distractors"],
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
        # # acc = sum(classify(task, txt, k_shot) == lab for txt, lab in task.test_pairs()) / len(task.test_pairs())
        # mcq_resp, correct = articulate_rule_mcq(task)
        # acc=1.0
        # # rows.append(dict(Task=task.name, Accuracy=acc, Rule=rule))
        # rows.append(dict(Task=task.name, Accuracy=acc, mcq_resp=mcq_resp, correct=correct))
        # # print(f"{task.name:20}  acc={acc:5.1%}  {rule}")
        # print(f"{task.name:20}  acc={acc:5.1%}  {mcq_resp}, correct={correct}")
        faith_overall, faith_pos, faith_neg = faithfulness_score(task, k_shot)
        acc=1.0
        rows.append({"Task": task.name,
                    "Accuracy": acc,
                    "Faith": faith_overall,
                    "Faith_pos": faith_pos,
                    "Faith_neg": faith_neg,
                    "Rule": task.rule})
        print(f"{task.name:20}  acc={acc:5.1%}  faith={faith_overall:4.1%} faith_pos={faith_pos:4.1%} faith_neg={faith_neg:4.1%}  {task.rule}")
    # print("\n=== Summary COT ===")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    run_experiments()
