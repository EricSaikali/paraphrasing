import random

import numpy as np
import torch
from datasets import load_dataset, ClassLabel, DatasetDict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_quora_questions(sentence):
    ids = sentence["questions"]["id"]
    txts = sentence["questions"]["text"]
    return {
        "question1_id": ids[0],
        "question2_id": ids[1],
        "question1": txts[0],
        "question2": txts[1],
    }


def load_quora_dataset(seed, test_proportion, validation_proportion):
    dataset = (load_dataset("quora-competitions/quora", trust_remote_code=True)
    .map(split_quora_questions,
         remove_columns=['questions']).cast_column(
        "is_duplicate",
        ClassLabel(names=["not_duplicate", "duplicate"])
    ))

    train_testvalid_dataset = dataset['train'].train_test_split(test_size=test_proportion, seed=seed,
                                                                stratify_by_column='is_duplicate')

    test_valid_dataset = train_testvalid_dataset['test'].train_test_split(test_size=validation_proportion, seed=seed,
                                                                          stratify_by_column='is_duplicate')
    quora_dataset = DatasetDict({
        'train': train_testvalid_dataset['train'],
        'test': test_valid_dataset['test'],
        'valid': test_valid_dataset['train']})

    return quora_dataset


def find_device(use_gpu: bool):
    """
    Function which determines the device to use.
    :param use_gpu: True if we want to use GPU, False otherwise.
    :return: Torch device to use.
    """
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")  # Use MPS (Apple Silicon)
    else:
        device = torch.device("cpu")
    return device


def display_paraphrasing_results(sources, references, paraphrases, displayed_num=5):
    assert len(paraphrases) == len(sources) and len(paraphrases) == len(references)

    for source, reference, paraphrase in zip(sources[:displayed_num],
                                             references[:displayed_num],
                                             paraphrases[:displayed_num]):
        print(f"Paraphrase: {paraphrase}")
        print(f"Source: {source}")
        print(f"Reference: {reference}")
        print(f"\n")


def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
