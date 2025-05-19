import sentence_transformers
import torch


class RewardModel(torch.nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.sentence_encoder = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = torch.nn.Linear(3 * 384, 1)
        self.device = device

    def forward(self, original, generated):

        original_emb = self.sentence_encoder.encode(original, convert_to_tensor=True).to(self.device)
        generated_emb = self.sentence_encoder.encode(generated, convert_to_tensor=True).to(self.device)

        diff = torch.abs(original_emb - generated_emb)
        features = torch.cat([original_emb, generated_emb, diff], dim=-1)

        score = self.classifier(features)
        return torch.sigmoid(score)