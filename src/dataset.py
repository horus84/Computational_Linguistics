import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# Dataset class (from SG6z_vqJzICK)
class SeqDataset(Dataset):
    def __init__(self, df, sign2idx, trans2idx, max_in=32, max_out=16, target_col='translit_seq'):
        self.data = df
        self.sign2idx = sign2idx
        self.trans2idx = trans2idx
        self.max_in = max_in
        self.max_out = max_out
        self.target_col = target_col # Added flexibility for target column
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sseq = self.data.loc[idx,'sign_seq']
        tseq = self.data.loc[idx, self.target_col] # Use target_col here

        # Ensure tseq is a list of tokens
        if isinstance(tseq, str):
             tseq = tseq.split()
        elif not isinstance(tseq, list):
             tseq = [] # Handle unexpected types gracefully


        ix = [self.sign2idx.get(s,0) for s in sseq][:self.max_in]
        ix += [0]*(self.max_in-len(ix))

        # Ensure tokens in tseq are in the vocabulary before lookup
        ox = [self.trans2idx.get('<sos>', 1)] + [self.trans2idx.get(t, self.trans2idx.get('<pad>', 0)) for t in tseq][:self.max_out] + [self.trans2idx.get('<eos>', 2)] # Use .get with default
        ox += [self.trans2idx.get('<pad>', 0)]*(self.max_out+2-len(ox)) # Use .get with default for padding

        return torch.tensor(ix), torch.tensor(ox)
