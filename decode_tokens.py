#!/usr/bin/env python3
"""Decode Qwen token IDs to text"""

from transformers import AutoTokenizer

# Your token IDs
token_ids = [118086, 100294, 100190, 3837, 99466, 52801, 6313, 104198,
             48, 16948, 3837, 67071, 102661, 107458, 114257, 100362,
             9370, 105292, 100482, 31548, 17340, 1773, 104198, 48,
             16948, 9370, 118086, 100294, 100190, 100653, 3837, 97611,
             101419, 105792, 118086, 100294, 100190, 1773, 104198, 48,
             16948, 9370, 118086, 100294, 100190, 100653, 3837, 97611,
             101419, 105792, 118086, 100294, 100190, 8997, 118086, 100294,
             100190, 20412, 48, 16948, 9370, 118086, 100294, 100190,
             100653, 3837, 97611, 101419, 105792, 118086, 100294, 100190,
             8997, 118086, 100294, 100190, 20412, 48, 16948, 9370,
             118086, 100294, 100190, 100653, 3837, 97611, 101419, 105792,
             118086, 100294, 100190, 1773, 151643]

# Load Qwen tokenizer (using a common Qwen model)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Decode the token IDs
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

print("Decoded text:")
print(repr(decoded_text))
print("\nReadable text:")
print(decoded_text)