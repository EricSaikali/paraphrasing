from torch.utils.data import Dataset


class QuoraDataset(Dataset):
    def __init__(self, hf_dataset):
        self.question1 = hf_dataset['question1']
        self.question2 = hf_dataset['question2']
        self.labels = hf_dataset['is_duplicate']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.question1[idx], self.question2[idx], self.labels[idx]
