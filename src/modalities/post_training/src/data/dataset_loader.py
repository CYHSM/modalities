import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class DatasetConfig:
    name: str
    split: str = "train"
    weight: float = 1.0
    filters: Dict[str, Any] = None


class InstructionDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template_style: str = "chatml",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_style = template_style

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Apply template
        formatted_text = self.apply_template(sample["instruction"], sample["response"])

        # Tokenize
        encoding = self.tokenizer(
            formatted_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )

        # Create labels (mask instruction part for loss calculation)
        labels = encoding["input_ids"].clone()

        # Find response start position and mask instruction
        response_start_token = self.get_response_start_token()
        response_start_idx = self.find_substring_position(encoding["input_ids"][0], response_start_token)

        if response_start_idx > 0:
            labels[0, :response_start_idx] = -100

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0],
        }

    def apply_template(self, instruction: str, response: str) -> str:
        templates = {
            "chatml": "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
            "alpaca": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
            "vicuna": f"USER: {instruction}\nASSISTANT: {response}",
            "llama2": f"<s>[INST] {instruction} [/INST] {response}</s>",
            "plain": f"Human: {instruction}\n\nAssistant: {response}",
        }
        return templates.get(self.template_style, templates["plain"])

    def get_response_start_token(self) -> List[int]:
        markers = {
            "chatml": "<|im_start|>assistant",
            "alpaca": "### Response:",
            "vicuna": "ASSISTANT:",
            "llama2": "[/INST]",
            "plain": "Assistant:",
        }
        marker = markers.get(self.template_style, markers["plain"])
        return self.tokenizer.encode(marker, add_special_tokens=False)

    def find_substring_position(self, tokens: torch.Tensor, substring: List[int]) -> int:
        tokens_list = tokens.tolist()
        for i in range(len(tokens_list) - len(substring) + 1):
            if tokens_list[i : i + len(substring)] == substring:
                return i + len(substring)
        return -1


class DatasetLoader:
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.datasets_config = config["data"]["datasets"]
        self.quality_filters = config["data"].get("quality_filters", {})

    def load_and_process_datasets(self) -> InstructionDataset:
        all_samples = []

        for dataset_cfg in self.datasets_config:
            print(f"Loading dataset: {dataset_cfg['name']}")
            samples = self.load_single_dataset(dataset_cfg)

            # Apply filters
            if dataset_cfg.get("filters"):
                samples = self.apply_filters(samples, dataset_cfg["filters"])

            # Apply quality filters
            if self.quality_filters:
                samples = self.apply_quality_filters(samples)

            # Sample based on weight
            num_samples = min(
                len(samples), int(dataset_cfg["weight"] * self.config["data"].get("max_samples_per_dataset", 50000))
            )
            samples = random.sample(samples, num_samples)

            all_samples.extend(samples)
            print(f"Added {len(samples)} samples from {dataset_cfg['name']}")

        # Shuffle all samples
        random.shuffle(all_samples)

        # Create dataset
        return InstructionDataset(
            all_samples,
            self.tokenizer,
            self.config["data"]["prompt"]["max_length"],
            self.config["data"]["prompt"]["template_style"],
        )

    def load_single_dataset(self, dataset_cfg: Dict) -> List[Dict]:
        dataset = load_dataset(dataset_cfg["name"], split=dataset_cfg.get("split", "train"))

        samples = []
        for item in dataset:
            # Handle different dataset formats
            instruction, response = self.extract_instruction_response(item, dataset_cfg["name"])
            if instruction and response:
                samples.append({"instruction": instruction, "response": response, "source": dataset_cfg["name"]})

        return samples

    def extract_instruction_response(self, item: Dict, dataset_name: str) -> tuple:
        # Handle different dataset formats

        # Check for common formats
        if "instruction" in item and "output" in item:
            return item["instruction"], item["output"]
        elif "instruction" in item and "response" in item:
            return item["instruction"], item["response"]
        elif "prompt" in item and "response" in item:
            return item["prompt"], item["response"]
        elif "question" in item and "answer" in item:
            return item["question"], item["answer"]
        elif "conversations" in item:
            # Handle conversation format
            convs = item["conversations"]
            if len(convs) >= 2:
                return convs[0], convs[1]
        elif "messages" in item:
            # Handle messages format
            msgs = item["messages"]
            if len(msgs) >= 2:
                return msgs[0].get("content", ""), msgs[1].get("content", "")

        return None, None

    def apply_filters(self, samples: List[Dict], filters: Dict) -> List[Dict]:
        filtered_samples = []

        for sample in samples:
            # Tokenize to check lengths
            instruction_tokens = self.tokenizer.encode(sample["instruction"])
            response_tokens = self.tokenizer.encode(sample["response"])

            # Check all filter conditions
            if (
                filters.get("min_instruction_tokens", 0)
                <= len(instruction_tokens)
                <= filters.get("max_instruction_tokens", float("inf"))
            ):
                if (
                    filters.get("min_response_tokens", 0)
                    <= len(response_tokens)
                    <= filters.get("max_response_tokens", float("inf"))
                ):
                    total_tokens = len(instruction_tokens) + len(response_tokens)
                    if (
                        filters.get("min_total_tokens", 0)
                        <= total_tokens
                        <= filters.get("max_total_tokens", float("inf"))
                    ):
                        filtered_samples.append(sample)

        return filtered_samples

    def apply_quality_filters(self, samples: List[Dict]) -> List[Dict]:
        filtered_samples = []

        for sample in samples:
            text = sample["instruction"] + " " + sample["response"]

            # Check alphabetic ratio
            if self.quality_filters.get("min_alphabetic_ratio"):
                alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
                if alpha_ratio < self.quality_filters["min_alphabetic_ratio"]:
                    continue

            # Check repetition
            if self.quality_filters.get("max_repetition_ratio"):
                if self.has_high_repetition(text, self.quality_filters["max_repetition_ratio"]):
                    continue

            # Check code content
            if self.quality_filters.get("remove_code_heavy"):
                if self.is_code_heavy(text):
                    continue

            # Check math content
            if self.quality_filters.get("remove_math_heavy"):
                if self.is_math_heavy(text):
                    continue

            filtered_samples.append(sample)

        # Diversity sampling
        if self.quality_filters.get("diversity_sampling"):
            filtered_samples = self.diversity_sample(filtered_samples)

        return filtered_samples

    def has_high_repetition(self, text: str, threshold: float) -> bool:
        words = text.split()
        if len(words) < 10:
            return False

        # Check 3-gram repetition
        ngrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        if total_ngrams > 0:
            repetition_ratio = 1 - (unique_ngrams / total_ngrams)
            return repetition_ratio > threshold
        return False

    def is_code_heavy(self, text: str) -> bool:
        code_indicators = ["def ", "class ", "import ", "function", "```", "var ", "const ", "let "]
        code_count = sum(1 for indicator in code_indicators if indicator in text)
        return code_count > 5 or "```" in text

    def is_math_heavy(self, text: str) -> bool:
        math_indicators = ["∫", "∑", "∏", "√", "∂", "∇", "∆", "dx", "dy"]
        math_count = sum(1 for indicator in math_indicators if indicator in text)
        return math_count > 3

    def diversity_sample(self, samples: List[Dict], num_clusters: int = 100) -> List[Dict]:
        if len(samples) <= num_clusters:
            return samples

        # Simple diversity sampling based on instruction embeddings
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        instructions = [s["instruction"] for s in samples]

        # Create TF-IDF embeddings
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        embeddings = vectorizer.fit_transform(instructions)

        # Cluster
        kmeans = KMeans(n_clusters=min(num_clusters, len(samples)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Sample from each cluster
        diverse_samples = []
        for cluster_id in range(num_clusters):
            cluster_samples = [s for i, s in enumerate(samples) if clusters[i] == cluster_id]
            if cluster_samples:
                # Take best sample from each cluster (longest response)
                best_sample = max(cluster_samples, key=lambda x: len(x["response"]))
                diverse_samples.append(best_sample)

        return diverse_samples
