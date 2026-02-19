import re
import torch
import tiktoken

class SimpleTokeniserV2:
    def __init__(self, vocabulary):
        '''
        * token is key and idx is value
        * basically vocabulary, but the idx is key and token is value
        * V2 now handles unknown text. 
        '''
        self.str_to_int = vocabulary
        self.int_to_str = {integer: token for token, integer in vocabulary.items()} 
    
    def encode(self, text: str, output: bool = True) -> list[int]:
        '''
        1. Split into str into individual characters (remove spacing)
        2. Map with string_to_int vocabulary to output numbers.
        3. Outputs a list of integar mapped values.
        * Addition in V2: if the word which is to be encoded does not exist in the current vocabulary, then it will be replaced with <|unk|> token.
        '''
        split_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split_tokens = [item.strip() for item in split_tokens if item.strip()]
        split_tokens = [item if item in self.str_to_int else "<|unk|>" for item in split_tokens]
        ids = [self.str_to_int[integer] for integer in split_tokens]
        if (output):
            print(ids)
        return ids

    def decode(self, ids: list[int], output: bool = True) -> str:
        '''
        1. Get a list of integars to be converted to a string form.
        2. Map the integer idx from int_to_string to string.
        3. Outputs a singular str of each char mapped.
        '''
        text = " ".join([self.int_to_str[idx] for idx in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        if (output):
            print(text)
        return text
    
def generate_vocabulary(text: str, output: bool = False) -> dict[str, int]:
    '''
    1. Split each character into different tokens (this includes punctuation, !, e.t.c.)
    2. Creating unique tokens (creating a vocabulary for the model)
    3. Allocating each unique token to a numeric id. 
    '''
    all_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    unique_tokens = sorted(list(set(all_tokens)))
    unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocabulary = {token: idx for idx, token in enumerate(unique_tokens)}
    if output:
        print(vocabulary)
    return vocabulary

if __name__ == "__main__":
    
    # Reading Text Document
    text_dir = r'data\romeo_and_juliet.txt'
    with open(text_dir, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    vocabulary = generate_vocabulary(raw_text, True)
    tokeniser = SimpleTokeniserV2(vocabulary)
    tokeniser.decode(tokeniser.encode('Hello world'))
    print(len(tokeniser.encode(raw_text)))
