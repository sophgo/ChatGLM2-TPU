#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


import datetime
import math
import unittest
import torch
import random
import sys

from transformers import AutoTokenizer, AutoModel

CHATGLM2_PATH = "/workspace/chatglm2-6b"
def get_model_and_tokenizer():
    model = AutoModel.from_pretrained(CHATGLM2_PATH, trust_remote_code=True).float().cpu()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CHATGLM2_PATH, trust_remote_code=True)
    return model, tokenizer

def main():
    model, tokenizer = get_model_and_tokenizer()
    print("Question:")
    history = []
    s = sys.stdin.readline().strip()
    while s != 'exit':
        print("Answer:")
        response, history = model.chat(tokenizer, s, history=history)
        print(response)
        print("Question:")
        s = sys.stdin.readline().strip()

if __name__ == '__main__':
    main()