import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    
    '''
    Creating the sliding window dataset which contains the input and respective target lists. 
    '''

    def __init__(self, text, tokeniser, window_size, stride):
        
        self.input_ids = []
        self.target_ids = []
        token_ids = tokeniser.encode(text)

        for idx in range(0, len(token_ids) - window_size, stride):
            # check the image in the sliding window for saving inputs and targets in words document. 
            input_chunk = token_ids[idx:idx + window_size]
            target_chunk = token_ids[idx + 1:idx + window_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def load_data(text: str, batch_size:int = 4, window_size: int = 256, stride: int = 128, shuffle: bool = False, drop_lost: bool = True, num_of_workers: int = 0):
    '''
    Creating an interative wrapper for the DataSet class (derived for our data.)
    '''
    tokeniser = tiktoken.get_encoding('o200k_base')
    dataset = GPTDatasetV1(text, tokeniser, window_size, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_lost,
        num_workers = num_of_workers
    )
    return dataloader

def sliding_window(data: str, tiktokeniser: tiktoken.Encoding, window_size: int):
    for idx in range(1, window_size+1):
        window = data[:idx]
        desired = data[idx]
        print(f'[{tiktokeniser.decode(window)}]: {tiktokeniser.decode([desired])}')

if __name__ == "__main__":
    
    file_dir = 'data/romeo_and_juliet.txt'

    with open(file_dir, 'r', encoding='utf-8') as raw_file:
        raw_text = raw_file.read()

    # # Testing sliding window
    # tiktokenisor = tiktoken.get_encoding("o200k_base")
    # encoded_values = tiktokenisor.encode(raw_text)
    # sample_encoded_values = encoded_values[20:80]
    # context_size = 10
    # sliding_window(sample_encoded_values, tiktokenisor, context_size

    load_data = load_data(raw_text, batch_size=2, window_size=8, stride=3)
    iter_load_data = iter(load_data)
    print(f'First Iteration: {next(iter_load_data)}') # contains 2 tensors; 1st is the window with context, and second is the target_ids
    print(f'Second Iteration: {next(iter_load_data)}')
    print(f'Third Iteration: {next(iter_load_data)[0]}')


