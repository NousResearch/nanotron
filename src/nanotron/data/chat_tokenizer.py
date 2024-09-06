from typing import List, Tuple
from enum import Enum, auto

from transformers import AutoTokenizer


class ChatFormat(Enum):
    LLAMA3 = auto()
    CHATML = auto()


class ChatTokenizer:
    """
    The ChatTokenizer encodes a conversation applying the Llama3 Chat Template and returns the role (Either User or Assistant) of each token

    Args:
        tokenizer_name_or_path (str): A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.
    """

    def __init__(self, tokenizer_name_or_path: str, chat_format: ChatFormat = ChatFormat.LLAMA3):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self._chat_format = chat_format
        if chat_format == ChatFormat.LLAMA3:
            self._header_start = "<|start_header_id|>"
            self._header_end = "<|end_header_id|>\n\n"
            self._turn_end = "<|eot_id|>"
        elif chat_format == ChatFormat.CHATML:
            self._header_start = "<|im_start|>"
            self._header_end = "\n"
            self._turn_end = "<|im_end|>\n"

        # Add pad token if necessary
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self._turn_end})

    def __call__(self, conversation: List[dict]) -> Tuple[List[int], List[bool]]:
        """
        Applies the Llama3 chat template, encodes the conversation and returns the tokens along with a bool value for each token whether if the token belongs to the answer of the assistant or not to be able to just train on the assistant answers
        Args:
            conversation (List[dict]):  List of dicts where each dict contains the "from" key to specify the emisor del mensaje and the "value" key with the message.
                                        Same format as SlimOrca dataset with possible from values: "System", "human" and "gpt"
        Example:
            conversation: [ { "from": "system", "value": "You are an AI assistant that follows instruction extremely well. Help as much as you can."},
                          { "from": "human", "value": "Answer the following question: - number is 54 - debutteam is pittsburgh steelers - draftpick is 166 - birth date is 24 may 1982 - weight is 243 - nfl is wal475737 - debutyear is 2005 - finalteam is new york sentinels - statlabel is tackles sacks interceptions - heightin is 3 - statvalue is 9 0.0 1 - heightft is 6 - college is temple - birth place is pottstown , pennsylvania - draftyear is 2005 - position is linebacker - draftround is 5 - finalyear is 2009 Given the details above, guess who could this information be about.\nAnswer:"},
                          { "from": "gpt", "value": "The information provided seems to refer to Rian Wallace, a former NFL player."} ]

            After applying chat template:
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>human<|end_header_id|>

                Answer the following question: - number is 54 - debutteam is pittsburgh steelers - draftpick is 166 - birth date is 24 may 1982 - weight is 243 - nfl is wal475737 - debutyear is 2005 - finalteam is new york sentinels - statlabel is tackles sacks interceptions - heightin is 3 - statvalue is 9 0.0 1 - heightft is 6 - college is temple - birth place is pottstown , pennsylvania - draftyear is 2005 - position is linebacker - draftround is 5 - finalyear is 2009 Given the details above, guess who could this information be about.
                Answer:<|eot_id|><|start_header_id|>gpt<|end_header_id|>

                The information provided seems to refer to Rian Wallace, a former NFL player.<|eot_id|>
          returns:
              tokens (List[int]): A list of tokens e.g. [128000, 128006,   9125, 128007,    271,   2675,    527, ...,  12873,   2851,     13, 128009, 128001]
              is_completitions (List[bool]): A list of bools whether the tokens belong to the assistant answer or not e.g. [False, False, False, ..., False,  True,  True,  True,  True]
        """
        tokens = []
        # Append <|begin_of_text|>
        tokens.append(self.tokenizer.bos_token_id)
        is_completitions = [False] * len(tokens)

        for message in conversation:
            message_tokens, message_completitions = self.encode_message(message)
            tokens.extend(message_tokens)
            is_completitions.extend(message_completitions)

        # Append <|end_of_text|> token
        tokens.extend(self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False))
        is_completitions.append(
            False
        )  # NOTE(tj.solergibert) No need to predict <|end_of_text|> token from <|eot_id|> token

        return tokens, is_completitions

    def encode_message(self, message: dict) -> Tuple[List[int], List[int]]:
        # NOTE(tj.solergibert) The "from", "value", "gpt" keys are from SlimOrca Dataset. Llama3 HF Pretrained tokenizer uses another ones. We should stick to a
        # single format and document it properly rather than supporting multiple formats, as each DATASET will need a different
        # ChatTokenizer and the idea is that all Datasets share the same ChatTokenizer

        role, is_input = self._get_role(message)

        # Encode header
        tokens = self.tokenizer.encode(
            f"{self._header_start}{role}{self._header_end}", add_special_tokens=False
        )
        is_completitions = [False] * len(tokens)

        # Encode message
        tokens.extend(self.tokenizer.encode(message["value"].strip(), add_special_tokens=False))

        # Append <|eot_id|> token
        tokens.extend(self.tokenizer.encode(self._turn_end, add_special_tokens=False))

        # True if token belongs to assistant answer, False otherwise
        is_completitions.extend([not is_input] * (len(tokens) - len(is_completitions)))

        return tokens, is_completitions

    def _get_role(self, message: dict) -> Tuple[str, bool]:
        """
            Return the canonical role for a given message, as well as if its value
            should be considered input (and therefore not trained on)
        """
        role = message["from"]
        if role == "gpt" or role == "assistant":
            return "assistant", False
        elif role == "summary" or role == "author":
            return role, False
        elif role == "human":
            return "user", True
        else:
            return role, True