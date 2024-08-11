from typing import List

import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from nanotron.data.chat_tokenizer import ChatTokenizer
from nanotron.data.collator import (
    build_labels,
    build_labels_completions_only,
    build_position_ids,
    build_position_ids_dummy,
)
from nanotron import logging
from nanotron.logging import log_rank
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

logger = logging.get_logger(__name__)


class ChatDataset(IterableDataset):
    """
    Chat Dataset for training models with:
        1. Padding-Free Packing
        2. No cross-contamination between packed samples
        3. Train on completitions only
    Args:
        dataset_path (str): Path to the dataset in the file system. If provided, data will be loaded from this path instead of downloaded.
        tokenizer_name_or_path (str): Path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.
        seq_len (int): max sequence length
        train_on_completions_only (bool): Whether to just train on completitions or not. To be deleted
        remove_cross_attention (bool): Whether to just attend to the tokens from the same sample or to all (Vanilla mechanism). To be deleted
        split (str): Split of the dataset to train on
        conversation_column_name (str): Column name of the dataset containing the conversations
        dp_rank (int): rank of the current data parallel process
        dp_ranks_size (int): number of data parallel processes participating in training
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer_name_or_path,
        sequence_length: int,
        conversation_column_name: str,
        train_on_completions_only: bool = True,
        remove_cross_attention: bool = True,
        pack_samples: bool = True,
        split: str = "train",
        dp_rank: int = 0,
        dp_ranks_size: int = 1,
        sp_ranks_size: int = 1,
        skip_num_samples: int = None,  # TODO(tj.solergibert) Delete, check later comment
        seed: int = 1234,
    ) -> None:

        # WARN(tj.solergibert) Currently we DON'T support recovering training from a interruption. Check the following TODOs
        # TODO(tj.solergibert) Support checkpointing for resuming training. We have to store the number of consumed samples from the dataset (Which is different from the number of steps) and the BUFFERS.
        #       skip_num_samples will fail, as it's computed with the number of steps and as we are packing sequences we might have consumed MORE samples from the dataset
        # TODO(tj.solergibert) Support interleaving datasets

        self.dataset_path = dataset_path
        self.chat_tokenizer = ChatTokenizer(tokenizer_name_or_path)
        self.sequence_length = sequence_length
        self.conversation_column_name = conversation_column_name
        self.skip_num_samples = skip_num_samples
        self.seed = seed
        self.pack_samples = pack_samples
        self.sp_chunks = sp_ranks_size * 2 if sp_ranks_size > 1 else 1

        # Load, split and shuffle dataset
        self.dataset = load_dataset(dataset_path, split=split, streaming=True)
        self.dataset = split_dataset_by_node(self.dataset, dp_rank, dp_ranks_size)
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10_000)

        # TODO(tj.solergibert) Delete (debug), just 4 switching the training only on completitions setting
        if train_on_completions_only:
            self.create_labels = build_labels_completions_only
        else:
            self.create_labels = build_labels

        # TODO Delete (debug), just 4 switching the remove cross-attention setting
        if remove_cross_attention:
            self.create_position_ids = build_position_ids
        else:
            self.create_position_ids = build_position_ids_dummy

        # TODO(tj.solergibert) Delete (debug)
        self.debug_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)  # TODO delete debug
        self.debug_tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['from'] + '<|end_header_id|>\n\n'+ message['value'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}"

    def __iter__(self):
        max_buffer_token_len = 1 + self.sequence_length
        buffer_tokens: List[int] = []
        buffer_is_completition: List[int] = []
        buffer_lengths: List[int] = []
        num_samples = 0

        while True:
            for sample in iter(self.dataset):
                tokens, is_completition = self.chat_tokenizer(sample[self.conversation_column_name])

                # TODO(tj.solergibert) Delete (debug). Check if HF apply_chat_template produces the same result as ChatTokenizer
                # The [:-1] of tokens is because apply chat template doesn't adds eos (NOT eot) token
                assert (
                    self.debug_tokenizer.apply_chat_template(sample["conversations"]) == tokens[:-1]
                ), f'{self.debug_tokenizer.apply_chat_template(sample["conversations"])}\n\n{tokens[:-1]}'

                if self.pack_samples:
                    buffer_tokens.extend(tokens)
                    buffer_is_completition.extend(is_completition)
                    buffer_lengths.append(len(tokens))

                    if len(buffer_tokens) > max_buffer_token_len:  # Can't pack more samples, yield
                        # Pop last sample from buffers
                        sample_tokens = buffer_tokens[: -len(tokens)]
                        sample_completitions = buffer_is_completition[: -len(tokens)]
                        sample_lengths = buffer_lengths[:-1]

                        # TODO(tj.solergibert) Delete (debug)
                        assert len(sample_tokens) == len(sample_completitions) == sum(sample_lengths)

                        # Reset tokens buffers
                        buffer_tokens = tokens.copy()
                        buffer_is_completition = is_completition.copy()
                        buffer_lengths = [len(tokens)]

                        # TODO(tj.solergibert) Delete (debug), just 4 switching the training only on completitions setting
                        sample_completitions = self.create_labels(sample_tokens, sample_completitions)

                        # TODO(tj.solergibert) Delete (debug), just 4 switching the remove cross-attention setting
                        position_ids = self.create_position_ids(sample_lengths, self.sequence_length)

                        # TODO(tj.solergibert) Delete (debug)
                        # assert len(sample_tokens) <= max_buffer_token_len

                        yield {
                            "input_ids": np.array(sample_tokens, dtype=np.int32),
                            "label_mask": np.array([1 if x else 0 for x in sample_completitions], dtype=np.int32),
                            "position_ids": position_ids,
                        }
                else:
                    # TODO(tj.solergibert) Delete (debug), just 4 switching the training only on completitions setting
                    is_completition = self.create_labels(tokens, is_completition)
                    input_mask = ([1] * len(tokens))

                    rem = len(tokens) % self.sp_chunks
                    if rem != 0:
                        pad_amount = self.sp_chunks - rem
                        tokens.extend([self.chat_tokenizer.tokenizer.pad_token_id] * pad_amount)
                        is_completition.extend([False] * pad_amount)
                        input_mask.extend([0] * pad_amount)

                    if self.sp_chunks > 1:
                        # sequence needs to be of length (closest multiple of 2 * sp_pg.size()) + 1
                        # + 1 is so we have (closest multiple of 2 * sp_pg.size()) after shifting by one to get causal prediction
                        tokens.append(self.chat_tokenizer.tokenizer.pad_token_id)
                        is_completition.append(False)
                        input_mask.append(0)

                    assert len(tokens) == len(input_mask)
                    assert len(input_mask) == len(is_completition)
                    label_mask = [1 if x else 0 for x in is_completition]

                    yield {
                        "input_ids": np.array(tokens, dtype=np.int32),
                        "input_mask": np.array(input_mask, dtype=np.int32),
                        "label_mask": np.array(label_mask, dtype=np.int32),
                    }
                    num_samples += 1

            # TODO(tj.solergibert) Change for log_rank (log_rank is problematic with JupyterNB)
            print("Consumed all samples, dataset is being re-looped.")