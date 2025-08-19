import random
from typing import List


class PromptTemplates:
    def __init__(self, template_style: str = "chatml", use_variations: bool = True):
        self.template_style = template_style
        self.use_variations = use_variations

        # Define template variations for diversity
        self.instruction_prefixes = [
            "Please",
            "Could you",
            "I need you to",
            "Help me",
            "Can you",
            "",
        ]

        self.instruction_suffixes = [
            "Provide a detailed explanation.",
            "Be thorough in your response.",
            "Explain step by step.",
            "Give me a comprehensive answer.",
            "",
        ]

    def augment_instruction(self, instruction: str) -> str:
        """Add variations to instructions for diversity"""
        if not self.use_variations or random.random() > 0.3:
            return instruction

        prefix = random.choice(self.instruction_prefixes)
        suffix = random.choice(self.instruction_suffixes)

        augmented = instruction
        if prefix and not instruction.lower().startswith(prefix.lower()):
            augmented = f"{prefix} {augmented.lower()}"
        if suffix and not any(augmented.endswith(s) for s in [".", "?", "!"]):
            augmented = f"{augmented}. {suffix}"

        return augmented

    def create_system_prompts(self) -> List[str]:
        """Generate diverse system prompts"""
        return [
            "You are a helpful AI assistant.",
            "You are an AI assistant created to be helpful, harmless, and honest.",
            "You are a knowledgeable assistant ready to help with any questions.",
            "You are an intelligent AI designed to assist users effectively.",
            "You are a capable and friendly AI assistant.",
        ]

    def format_for_training(self, instruction: str, response: str, add_variations: bool = True) -> str:
        """Format instruction-response pair for training"""
        if add_variations:
            instruction = self.augment_instruction(instruction)

        return self.apply_template(instruction, response)

    def apply_template(self, instruction: str, response: str) -> str:
        """Apply the selected template style"""
        templates = {
            "chatml": self._chatml_template,
            "alpaca": self._alpaca_template,
            "vicuna": self._vicuna_template,
            "llama2": self._llama2_template,
            "plain": self._plain_template,
        }

        template_func = templates.get(self.template_style, self._plain_template)
        return template_func(instruction, response)

    def _chatml_template(self, instruction: str, response: str) -> str:
        system = random.choice(self.create_system_prompts()) if self.use_variations else self.create_system_prompts()[0]
        ret_value = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user"
            "\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        )
        return ret_value

    def _alpaca_template(self, instruction: str, response: str) -> str:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def _vicuna_template(self, instruction: str, response: str) -> str:
        return f"USER: {instruction}\nASSISTANT: {response}"

    def _llama2_template(self, instruction: str, response: str) -> str:
        return f"<s>[INST] {instruction} [/INST] {response}</s>"

    def _plain_template(self, instruction: str, response: str) -> str:
        return f"Human: {instruction}\n\nAssistant: {response}"
