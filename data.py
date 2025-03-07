import re

def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', book_text))

    filename_cleaned = f"{filename}_cleaned.txt"
    print(filename_cleaned, len(cleaned_text), "characters")

    with open(filename_cleaned, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)


import util
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = util.tokenizer.encode(txt)

        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def get_train_loader(filename, max_length = 32, stride = 4):
    with open(filename, 'r', encoding='utf-8-sig') as file: # remove BOM with -sig        
        txt = file.read()

    dataset = MyDataset(txt, max_length = max_length, stride = stride)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    return train_loader

