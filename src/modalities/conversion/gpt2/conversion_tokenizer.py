import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterable

from transformers import LlamaTokenizer

from modalities.tokenization.tokenizer_wrapper import PreTrainedSPTokenizer


def convert_tokenizer(tokenizer_model_path: str, output_dir: str):
    """
    Converts a SentencePiece tokenizer to a Huggingface tokenizer, ensuring
    special tokens are correctly configured.

    Args:
        tokenizer_model_path (str): Path to the SentencePiece tokenizer model file.
        output_dir (str): Path to the directory where the converted tokenizer will be saved.
    Returns:
        tuple[int, int, int, int]: The actual bos_token_id, eos_token_id, pad_token_id and
                                   unk_token_id of the tokenizer. Note, that these are not
                                   set in the transformers part of the created tokenizer.
                                   Only in the wrapped SentencePiece tokenizer.
    """
    print("Loading source SentencePiece tokenizer...")
    sp_tokenizer_wrapper = PreTrainedSPTokenizer(tokenizer_model_path)
    sp_model = sp_tokenizer_wrapper.tokenizer

    # 1. Get all special tokens and their IDs from the source model
    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()
    unk_id = sp_model.unk_id()
    pad_id = sp_model.pad_id()  # Often -1 if not set

    bos_token = sp_model.id_to_piece(bos_id) if bos_id != -1 else "<s>"
    eos_token = sp_model.id_to_piece(eos_id) if eos_id != -1 else "</s>"
    unk_token = sp_model.id_to_piece(unk_id) if unk_id != -1 else "<unk>"

    # 2. Handle the PAD token
    if pad_id == -1:
        print("Warning: PAD token not set in source tokenizer. Using EOS token as PAD.")
        pad_id = eos_id
        pad_token = eos_token
    else:
        pad_token = sp_model.id_to_piece(pad_id)

    with _create_tokenizer_directory(tokenizer_model_path) as tokenizer_model_dir:
        print(f"Loading LlamaTokenizer from temporary directory: {tokenizer_model_dir}")
        hf_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model_dir, legacy=True)

    # 3. Set the special tokens on the LlamaTokenizer
    hf_tokenizer.add_special_tokens(
        {"bos_token": bos_token, "eos_token": eos_token, "unk_token": unk_token, "pad_token": pad_token}
    )

    # Also set the token_id attributes directly to ensure config is written correctly
    hf_tokenizer.bos_token_id = bos_id
    hf_tokenizer.eos_token_id = eos_id
    hf_tokenizer.unk_token_id = unk_id
    hf_tokenizer.pad_token_id = pad_id

    # 4. Save the fully configured tokenizer
    print(f"Saving configured tokenizer to: {output_dir}")
    hf_tokenizer.save_pretrained(output_dir)
    print("Conversion complete.")
    print(f"  BOS: '{bos_token}' (ID: {bos_id})")
    print(f"  EOS: '{eos_token}' (ID: {eos_id})")
    print(f"  UNK: '{unk_token}' (ID: {unk_id})")
    print(f"  PAD: '{pad_token}' (ID: {pad_id})")

    return bos_id, eos_id, pad_id, unk_id


@contextmanager
def _create_tokenizer_directory(tokenizer_model_path: str) -> Iterable[str]:
    """Copies the tokenizer model to a temporary directory and yields the path to the model.
       The model is moved to a temporary directory because the from_pretrained method of
       the LlamaTokenizer class requires a directory path instead of a file path from transformers v5 on.
       When the returned iterator is exhausted, the temporary directory is deleted.

    Args:
        tokenizer_model_path (str): Path to the tokenizer model file.

    Yields:
        Iterable[str]: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy(tokenizer_model_path, f"{temp_dir}/tokenizer.model")
        yield temp_dir
