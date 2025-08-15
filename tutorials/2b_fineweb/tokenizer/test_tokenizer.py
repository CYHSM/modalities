#!/usr/bin/env python3
"""
Test script for SentencePiece tokenizer from local file
"""

import os
import sys

import sentencepiece as spm


def test_sentencepiece_from_path(model_path):
    """Test SentencePiece tokenizer from local model file"""
    print(f"=== Testing SentencePiece from: {model_path} ===")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        # Load the SentencePiece model
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        # Basic info
        print(f"Vocabulary size: {sp.vocab_size()}")
        print(f"BOS token: {sp.bos_id()} -> '{sp.id_to_piece(sp.bos_id())}'")
        print(f"EOS token: {sp.eos_id()} -> '{sp.id_to_piece(sp.eos_id())}'")
        print(f"UNK token: {sp.unk_id()} -> '{sp.id_to_piece(sp.unk_id())}'")
        print(
            f"PAD token: {sp.pad_id()}"
            + (f" -> '{sp.id_to_piece(sp.pad_id())}'" if sp.pad_id() != -1 else " (not set)")
        )
        print()

        return sp

    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")
        return None


def test_tokenization(sp):
    """Test tokenization with various inputs"""
    if sp is None:
        return

    print("=== Tokenization Tests ===")

    # Test sentences
    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!",
        "这是一个中文句子。",  # Chinese
        "Bonjour, comment allez-vous?",  # French
        "¿Cómo estás hoy?",  # Spanish
        "def hello_world():\n    print('Hello, World!')",  # Code
        "Artificial Intelligence and Natural Language Processing",
        "tokenization subword splitting",
    ]

    for text in test_texts:
        # Different encoding methods
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        decoded = sp.decode_pieces(pieces)
        decoded_ids = sp.decode_ids(ids)

        print(f"Original: '{text}'")
        print(f"Pieces: {pieces}")
        print(f"IDs: {ids}")
        print(f"Decoded from pieces: '{decoded}'")
        print(f"Decoded from IDs: '{decoded_ids}'")
        print(f"Token count: {len(pieces)}")
        print(f"Perfect reconstruction: {text == decoded == decoded_ids}")
        print("-" * 80)


def analyze_vocabulary(sp, sample_size=20):
    """Analyze vocabulary structure"""
    if sp is None:
        return

    print(f"=== Vocabulary Analysis (showing first {sample_size} tokens) ===")

    # Show first N tokens
    print("First tokens in vocabulary:")
    for i in range(min(sample_size, sp.vocab_size())):
        piece = sp.id_to_piece(i)
        print(f"ID {i:4d}: '{piece}'")

    print(f"\nLast {sample_size} tokens in vocabulary:")
    for i in range(max(0, sp.vocab_size() - sample_size), sp.vocab_size()):
        piece = sp.id_to_piece(i)
        print(f"ID {i:4d}: '{piece}'")


def test_edge_cases(sp):
    """Test edge cases and special scenarios"""
    if sp is None:
        return

    print("=== Edge Cases ===")

    edge_cases = [
        "",  # Empty string
        " ",  # Single space
        "a",  # Single character
        "aaaaaaaaaaaa",  # Repeated characters
        "123456789",  # Numbers
        "!@#$%^&*()",  # Special characters
        "https://www.example.com",  # URL
        "user@email.com",  # Email
        "▁",  # SentencePiece space symbol
        "\n\t",  # Whitespace characters
    ]

    for text in edge_cases:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"'{repr(text)}' -> Pieces: {pieces} | IDs: {ids}")


def test_subword_behavior(sp):
    """Test how subwords are handled"""
    if sp is None:
        return

    print("=== Subword Behavior ===")

    # Test long/complex words
    words = [
        "tokenization",
        "untokenizable",
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconios",
        "artificialintelligence",
        "naturallanguageprocessing",
    ]

    for word in words:
        pieces = sp.encode_as_pieces(word)
        ids = sp.encode_as_ids(word)
        print(f"'{word}' ({len(word)} chars)")
        print(f"  -> {len(pieces)} pieces: {pieces}")
        print(f"  -> IDs: {ids}")
        print()


def compare_encoding_methods(sp):
    """Compare different encoding methods"""
    if sp is None:
        return

    print("=== Encoding Methods Comparison ===")

    test_text = "Hello world! This is a test sentence."

    # Different sampling methods
    print(f"Text: '{test_text}'")
    print()

    # Regular encoding
    pieces = sp.encode_as_pieces(test_text)
    print(f"Regular: {pieces}")

    # With different sampling parameters (if supported)
    # Some SentencePiece models support sampling
    sampled = sp.sample_encode_as_pieces(test_text, -1, 0.1)
    print(f"Sampled (alpha=0.1): {sampled}")

    # NBest encoding (if supported)
    nbest = sp.nbest_encode_as_pieces(test_text, 3)
    print(f"NBest (top 3): {nbest}")


def interactive_test(sp):
    """Interactive testing mode"""
    if sp is None:
        return

    print("\n=== Interactive Mode ===")
    print("Enter text to tokenize (or 'quit' to exit):")

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            pieces = sp.encode_as_pieces(user_input)
            ids = sp.encode_as_ids(user_input)
            decoded = sp.decode_pieces(pieces)

            print(f"Pieces ({len(pieces)}): {pieces}")
            print(f"IDs: {ids}")
            print(f"Decoded: '{decoded}'")
            print(f"Perfect match: {user_input == decoded}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_tokenizer.py <path_to_tokenizer.model>")
        print("\nExample:")
        print("  python test_tokenizer.py ./tokenizer.model")
        print("  python test_tokenizer.py /path/to/downloaded/tokenizer.model")
        sys.exit(1)

    model_path = sys.argv[1]

    print("SentencePiece Tokenizer Test")
    print("=" * 50)

    # Load tokenizer
    sp = test_sentencepiece_from_path(model_path)

    if sp is None:
        print("Failed to load tokenizer. Exiting.")
        sys.exit(1)

    # Run tests
    test_tokenization(sp)
    analyze_vocabulary(sp)
    test_edge_cases(sp)
    test_subword_behavior(sp)
    compare_encoding_methods(sp)

    # Optional interactive mode
    try_interactive = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
    if try_interactive in ["y", "yes"]:
        interactive_test(sp)

    print("\nTesting complete!")


if __name__ == "__main__":
    # Required package
    print("Make sure you have installed: pip install sentencepiece")
    print()

    main()
