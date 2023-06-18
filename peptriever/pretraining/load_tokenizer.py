from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def load_local_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=tokenizer.padding["pad_token"]
    )
    return fast_tokenizer
