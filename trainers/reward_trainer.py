import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.quora_dataset import QuoraDataset
from mlp_logger import MLPLogger
from models.reward_model import RewardModel
from utils import load_quora_dataset, set_seed


def train_reward_model(
        reward_model: RewardModel,
        train_dataloader,
        num_epochs=3,
        learning_rate=1e-4,
        print_every=100,
        device='cpu',
):
    mlp_logger = MLPLogger(print_every=print_every, save_dir='storage/')
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    num_samples = len(train_dataloader)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (original, other, labels) in enumerate(train_dataloader):
            score = reward_model(original, other)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
            loss = criterion(score, labels)

            predicted = score.round(decimals=0)
            current_accuracy = (predicted == labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            mlp_logger.log_iteration_loss_accuracy(current_loss,
                                                   current_accuracy,
                                                   epoch,
                                                   i)

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / num_samples}")

    return reward_model


def test_reward_model(reward_model: RewardModel, test_dataloader):
    reward_model.eval()
    with torch.no_grad():
        num_test_samples = len(test_dataloader)
        accuracy_sum = 0
        total = 0
        for questions1, questions2, labels in tqdm(test_dataloader, total=num_test_samples // reward_batch_size):
            scores = reward_model(questions1, questions2)
            predicted = scores.round(decimals=0).view(-1)
            labels = labels.view(-1)
            accuracy_sum += (predicted == labels).float().sum().item()
            total += labels.size(0)
    return accuracy_sum / total


if __name__ == '__main__':
    SEED = 42
    QUORA_TEST_PROPORTION = 0.2
    QUORA_VALID_PROPORTION = 0.5

    reward_batch_size = 128
    reward_num_epochs = 5
    reward_learning_rate = 1e-4
    reward_print_every = 100

    reward_model = RewardModel()
    reward_model.train()

    set_seed(SEED)
    quora_dataset = load_quora_dataset(SEED,
                                       QUORA_TEST_PROPORTION,
                                       QUORA_VALID_PROPORTION)

    train_dataset = QuoraDataset(quora_dataset['train'])
    valid_dataset = QuoraDataset(quora_dataset['valid'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=reward_batch_size,
                                  shuffle=True,
                                  drop_last=False)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=reward_batch_size,
                                  shuffle=False,
                                  drop_last=False)

    train_reward_model(
        reward_model,
        train_dataloader,
        num_epochs=reward_num_epochs,
        learning_rate=reward_learning_rate,
        print_every=reward_print_every)

    test_reward_model(reward_model, valid_dataloader)
