import os
import torch
from torch.utils.data import Dataset


# dataset for handelling sentences
class SentimentDataset(Dataset):
    def __init__(self, text_fns, tokenizer, max_length):
        self.text_fns = text_fns
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_fns)

    def __getitem__(self, idx):
        text_fn = self.text_fns[idx]
        label = 0
        if os.path.basename(os.path.dirname(text_fn)) == 'pos': label = 1
        with open(text_fn) as f:
            text = f.readlines()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class JokesDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.jokes = self.load_jokes(filepath)
        self.max_length = max_length

    def load_jokes(self, filepath):
        jokes = []
        with open(filepath, "r") as f:
            lines = f.readlines()
            for row in lines:
                joke = row['Joke']
                joke_split = joke.split()
                if len(joke_split) > 3:
                    first_three_words = " ".join(joke_split[:3])
                    rest_of_joke = " ".join(joke_split[3:])
                    jokes.append((first_three_words, rest_of_joke))
        return jokes

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        first_three_words, rest_of_joke = self.jokes[idx]
        
        # Tokenize inputs and outputs
        input_encodings = self.tokenizer(first_three_words, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        target_encodings = self.tokenizer(rest_of_joke, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        input_ids = input_encodings["input_ids"].squeeze()  # Remove batch dimension
        target_ids = target_encodings["input_ids"].squeeze()

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }
