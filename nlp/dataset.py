import os


# dataset for handelling sentences
class SentimentDataset(Dataset):
    def __init__(self, text_fns, tokenizer, max_length):
        self.text_fns = text_fns
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_fns)

    def __getitem__(self, idx):
        text_fn = self.texts[idx]
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