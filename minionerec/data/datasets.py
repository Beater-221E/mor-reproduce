"""
Official-commit dataset ports for MiniOneRec SFT/RL.

Source: MiniOneRec-official @ 0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed
Files: data.py (SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset,
       SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset)

Only tasks actually ConcatDataset'd in sft.py / rl.py are enabled by default.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, List

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class Tokenizer:
    """Official Tokenizer wrapper (data.py)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        # Qwen may not have bos; strip accidental specials
        while self.bos_id is not None and t and t[0] == self.bos_id:
            t = t[1:]
        while self.eos_id is not None and t and t[-1] == self.eos_id:
            t = t[:-1]
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class BaseDataset(Dataset):
    def __init__(self, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        super().__init__()
        self.data = None
        self.inputs = None
        if tokenizer is not None:
            self.tokenizer = Tokenizer(tokenizer)
        if seed is not None:
            random.seed(seed)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

    def __len__(self):
        return len(self.data) if self.inputs is None else len(self.inputs)

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data)), desc=type(self).__name__, leave=False):
            item = self.pre(i)
            if item is not None:
                inputs.append(item)
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


class CSVBaseDataset(BaseDataset):
    def __init__(self, train_file, sample=-1, seed=0, max_len=2048, category="", dedup=False, tokenizer=None, test=False):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)


class JSONBaseDataset(BaseDataset):
    def __init__(self, item_file=None, index_file=None, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        with open(item_file, "r", encoding="utf-8") as f:
            self.item_feat = json.load(f)
        with open(index_file, "r", encoding="utf-8") as f:
            self.indices = json.load(f)


class SidSFTDataset(CSVBaseDataset):
    """Generative recommendation: SID history -> next SID. Official sft.py train_data1."""

    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)
        self.get_inputs()

    def get_history(self, row):
        row = row.copy()
        hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
        history = ", ".join(hist)
        target_item = str(row["item_sid"])
        last = hist[-1] if hist else None
        return {
            "input": (
                f"The user has interacted with items {history} in chronological order. "
                f"Can you predict the next possible item that the user may expect?"
            ),
            "output": target_item + "\n",
            "history_str": ", ".join(hist),
            "dedup": target_item == last,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
            "task": "sid_sft",
        }


class SidItemFeatDataset(JSONBaseDataset):
    """
    SID-item feature alignment.
    Official commit enables ONLY title2sid and sid2title (NOT description).
    Source: data.py SidItemFeatDataset; sft.py train_data2.
    """

    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        super().__init__(item_file=item_file, index_file=index_file, tokenizer=tokenizer, max_len=max_len, test=test, category=category, seed=seed)
        self.sid2title = {}
        self.title2sid = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                title = self.item_feat[item_id]["title"]
                combined_sid = sids[0] + sids[1] + sids[2]
                self.sid2title[combined_sid] = title
                self.title2sid[title] = combined_sid
        self.data = []
        for sid, title in self.sid2title.items():
            self.data.append({"task": "sid2title", "input": sid, "output": title})
        for title, sid in self.title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        if self.tokenizer is not None:
            self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'What is the title of item "{data_point["input"]}"?'
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        # Leakage guard (official answer-leakage fix): title2sid prompt must not contain SID.
        if data_point["task"] == "title2sid":
            assert data_point["output"] not in prompt, "title2sid prompt leaked SID answer"
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        target = data_point["output"] + "\n"
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
            "task": data_point["task"],
        }


class FusionSeqRecDataset(BaseDataset):
    """
    Fusion sequence recommendation.
    Official commit: SID history -> title (description branch commented out).
    Source: data.py FusionSeqRecDataset; sft.py train_data3.
    """

    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)
        with open(item_file, "r", encoding="utf-8") as f:
            self.item_feat = json.load(f)
        with open(index_file, "r", encoding="utf-8") as f:
            self.indices = json.load(f)
        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.sid2title[combined_sid] = self.item_feat[item_id]["title"]
        self.get_inputs()

    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item"

    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def get_history(self, row):
        history_item_sid = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
        history_str = ", ".join(history_item_sid)
        target_sid = row["item_sid"]
        target_title = self.sid2title.get(target_sid, target_sid)
        last = history_item_sid[-1] if history_item_sid else None
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": target_sid == last,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history_data = self.get_history(self.data.iloc[idx])
        if self.dedup and history_data["dedup"]:
            return None
        prompt = self.generate_prompt_title(history_data["history_str"])
        target = history_data["target_title"] + "\n"
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        tokens = tokens + self.tokenizer.encode(formatted_prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
            "task": "fusion_seq_title",
        }


# --------------- RL prompt datasets (official rl.py ConcatDataset) ---------------


class SidDataset(CSVBaseDataset):
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def get_history(self, row):
        hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
        history = ", ".join(hist)
        history_str = "::".join(hist)
        target_item = str(row["item_sid"])
        last = hist[-1] if hist else None
        return {
            "input": (
                f"The user has interacted with items {history} in chronological order. "
                f"Can you predict the next possible item that the user may expect?"
            ),
            "output": target_item + "\n",
            "history_str": history_str,
            "dedup": target_item == last,
        }

    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        return {"prompt": prompt, "completion": target_item, "task": "sid_rec"}


class RLTitle2SidDataset(JSONBaseDataset):
    """Official: title2sid + description2sid. rl.py train_data2."""

    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)
        self.prompt2history = {}
        self.history2target = {}
        self.title2sid = {}
        self.description2sid = {}
        for item_id, sids in self.indices.items():
            if item_id not in self.item_feat or len(sids) < 3:
                continue
            title = self.item_feat[item_id]["title"]
            description = self.item_feat[item_id]["description"]
            if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
                try:
                    desc_list = eval(description)
                    description = desc_list[0] if desc_list else description
                except Exception:
                    pass
            combined_sid = sids[0] + sids[1] + sids[2]
            self.title2sid[title] = combined_sid
            self.description2sid[description] = combined_sid
        self.data = []
        for title, sid in self.title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})
        for description, sid in self.description2sid.items():
            self.data.append({"task": "description2sid", "input": description, "output": sid})
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'An item can be described as follows: "{data_point["input"]}". Which item is it describing?'
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def pre(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point["output"] + "\n"
        self.prompt2history[prompt] = data_point["input"]
        self.history2target[data_point["input"]] = target_item
        return {"prompt": prompt, "completion": target_item, "task": data_point["task"]}


class RLSeqTitle2SidDataset(CSVBaseDataset):
    """Official rl.py train_data3 (sample=10000 in entry)."""

    def __init__(self, train_file, sample=-1, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len=1024, category=category, dedup=dedup, tokenizer=None, test=False)
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def generate_prompt(self, inter_titles):
        return f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"

    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def get_history(self, row):
        history_item_title = eval(row["history_item_title"]) if isinstance(row["history_item_title"], str) else row["history_item_title"]
        inter_titles = ", ".join([f'"{title}"' for title in history_item_title])
        return {
            "inter_titles": inter_titles,
            "target_sid": row["item_sid"],
            "history_str": "::".join(history_item_title),
            "dedup": False,
        }

    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        prompt = self.generate_prompt(history_data["inter_titles"])
        target = history_data["target_sid"] + "\n"
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        self.prompt2history[formatted_prompt] = history_data["history_str"]
        self.history2target[history_data["history_str"]] = target
        return {"prompt": formatted_prompt, "completion": target, "task": "seq_title2sid"}


class EvalSidDataset(CSVBaseDataset):
    """Official evaluate.py EvalSidDataset."""

    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)
        self.get_inputs()

    def get_history(self, row):
        hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
        history = ", ".join(hist)
        target_item = str(row["item_sid"])
        return {
            "input": f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
            "output": target_item + "\n",
            "history_item_sid": hist,
            "item_sid": target_item,
        }

    def get_all(self):
        return [self.get_history(self.data.iloc[i]) for i in range(len(self.data))]

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
        }
