import matplotlib.pyplot as plt
import numpy as np
import time


class MLPLogger:
    def __init__(self, print_every, save_dir):
        self.iteration_losses = []
        self.iteration_accuracies = []
        self.epochs_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.print_every = print_every
        self.save_dir = save_dir
        self.start_time = time.time()
        self.save_images = False

    def log_iteration_loss_accuracy(self, loss, accuracy, epoch, it):
        self.iteration_losses.append(loss)
        self.iteration_accuracies.append(accuracy)
        if len(self.iteration_losses) % self.print_every == 0:
            avg_iteration_losses = np.mean(np.array(self.iteration_losses). \
                                 reshape(-1, self.print_every), axis=1)
            if self.save_images:
                plt.plot(range(1, len(self.iteration_losses), self.print_every), avg_iteration_losses)
                plt.xlim(1, len(self.iteration_losses))
                plt.xlabel('Number of batches')
                plt.ylabel(f'Loss (avg. every {self.print_every}) batches')
                plt.savefig(f'{self.save_dir}_iter_loss.png')
                plt.close()
            elapsed_time = time.time() - self.start_time
            print(f'[Epoch {epoch}]',
                  f'Avg. loss over batches {it - self.print_every + 1}-{it}: {avg_iteration_losses[-1]:.4f} Final accuracy over batches {it - self.print_every + 1}-{it}: {accuracy*100:.2f}% ({elapsed_time:.0f} secs)')

    def log_accuracy_losses(self, train_value, val_value, is_accuracy):
        if is_accuracy:
            self.train_accs.append(train_value)
            self.val_accs.append(val_value)
            train_values = self.train_accs
            val_values = self.val_accs
            img_name = '_accuracy.png'
            y_label = 'Accuracy'
        else:
            self.epochs_losses.append(train_value)
            self.val_losses.append(val_value)
            train_values = self.epochs_losses
            val_values = self.val_losses
            img_name = '_epoch_loss.png'
            y_label = 'Loss'

        if self.save_images:
            x = range(0, len(train_values))
            plt.plot(x, train_values, label='Train')
            plt.plot(x, val_values, label='Validation')
            plt.xlim(0, len(train_values))
            plt.legend()
            plt.xlabel('Number of epochs')
            plt.ylabel(y_label)
            plt.savefig(f'{self.save_dir}{img_name}')
            plt.close()



