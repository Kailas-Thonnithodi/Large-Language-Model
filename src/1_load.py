import re
import torch

class SimpleTokeniserV1:
    def __init__(self, vocabulary):
        '''
        * token is key and idx is value
        * basically vocabulary, but the idx is key and token is value
        '''
        self.str_to_int = vocabulary
        self.int_to_str = {integer: token for token, integer in vocabulary.items()} 
    
    def encode(self, text: str, output: bool = True) -> list[int]:
        '''
        1. Split into str into individual characters (remove spacing)
        2. Map with string_to_int vocabulary to output numbers.
        '''
        split_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split_tokens = [item.strip() for item in split_tokens if item.strip()]
        ids = [self.str_to_int[integer] for integer in split_tokens]
        if (output):
            print(ids)
        return ids

    def decode(self, ids: list[int], output: bool = True) -> str:
        '''
        1. Get a list of integars to be converted to a string form.
        2. Map the integer idx from int_to_string to string.
        '''
        text = " ".join([self.int_to_str[idx] for idx in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        if (output):
            print(text)
        return text
    
        
def generate_vocabulary(text: str) -> dict[str, int]:
    '''
    1. Split each character into different tokens (this includes punctuation, !, e.t.c.)
    2. Creating unique tokens (creating a vocabulary for the model)
    3. Allocating each unique token to a numeric id. 
    '''
    all_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    unique_tokens = sorted(set(all_tokens))
    return {token: idx for idx, token in enumerate(unique_tokens)}

if __name__ == "__main__":
    
    # Reading Text Document
    text_dir = r'data\romeo_and_juliet.txt'
    with open(text_dir, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    vocabulary = generate_vocabulary(raw_text)
    tokenised_vocabulary = SimpleTokeniserV1(vocabulary)
    tokenised_vocabulary.encode("The Tragedie of Romeo and Juliet Actus Primus. Scoena Prima. Enter Sampson and Gregory, with Swords and Bucklers, of the House of Capulet.")

