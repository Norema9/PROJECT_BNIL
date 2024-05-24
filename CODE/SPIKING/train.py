import torch
import torch.nn.functional as F
from snntorch import spikegen
from snntorch.functional import ce_rate_loss
from snntorch.functional import accuracy_rate
from tqdm import tqdm
from dataset import Data_loader, create_dataset
from snn import SMLP
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

HIDDEN_SIZE = 12
NUM_HIDDEN_LAYER = 1
SEED = 42
EVAL_STEP = 10
CHECKPOINT_PATH = None
NUM_EPOCH = 4
TIME_WINDOW = 25
BATCH_TRAIN_SIZE = 1
BATCH_TEST_SIZE = 1
LEARNING_RATE = 0.001

def main(data_directory):
    data_train_inputs, data_test_inputs, data_train_outputs, data_test_outputs, label_encoder = create_dataset(data_directory, SEED)
    input_dim = data_train_inputs.shape[1]
    save_dir = os.path.join("MODELS_SNN", f"{NUM_EPOCH}_epochs_{LEARNING_RATE}_learning_rate")
    os.makedirs(save_dir, exist_ok = True)

    train_loader = Data_loader(data_train_inputs, data_train_outputs, batch_size = BATCH_TRAIN_SIZE, time_window = TIME_WINDOW, shuffle = True)
    test_loader = Data_loader(data_test_inputs, data_test_outputs, batch_size = BATCH_TEST_SIZE, time_window = TIME_WINDOW, shuffle = False)

    model = SMLP(input_size = input_dim, hidden_size = HIDDEN_SIZE, output_size = 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = ce_rate_loss()

    best_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        train_losses = checkpoint['train_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCH):
        model.train()
        acc = 0
        total = 0
        total_loss = 0

        for data, targets in tqdm(train_loader):
            optimizer.zero_grad()
            spike_data = spikegen.rate(data, num_steps=TIME_WINDOW)
            spk, mem = model(spike_data)
            loss = loss_fn(spk.unsqueeze(1), targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            acc += accuracy_rate(spk.unsqueeze(1), targets) * targets.size(0)
            total += targets.size(0)

        accuracy = acc / total * 100
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        if (epoch + 1) % EVAL_STEP == 0:
            model.eval()
            acc_test = 0
            total_test = 0
            with torch.no_grad():
                for data, targets in tqdm(test_loader):
                    spike_data = spikegen.rate(data, num_steps=TIME_WINDOW)
                    spk, mem = model(spike_data)
                    acc_test += accuracy_rate(spk.unsqueeze(1), targets) * targets.size(0)
                    total_test += targets.size(0)

            test_accuracy = acc_test / total_test * 100
            val_accuracies.append(test_accuracy)

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies
                }, os.path.join(save_dir, 'best_model.pth'))

            print(f"Epoch {epoch+1}, CE Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Test Accuracy: {test_accuracy:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }, os.path.join(save_dir, 'checkpoint.pth'))

    df = pd.DataFrame({
        'epoch': list(range(1, NUM_EPOCH + 1)),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)

    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))

    results = {
        'final_train_loss': train_losses[-1],
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': best_acc,
        'configurations': {
            'hidden_size': HIDDEN_SIZE,
            'num_hidden_layer': NUM_HIDDEN_LAYER,
            'seed': SEED,
            'eval_step': EVAL_STEP,
            'num_epoch': NUM_EPOCH,
            'time_window': TIME_WINDOW,
            'batch_train_size': BATCH_TRAIN_SIZE,
            'batch_test_size': BATCH_TEST_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\02-Documents\00.ETUDES\00.ECOLE_D_INGE\00.CYCLE_ING_FORMATION_INIT\00.3EME_ANNEE_INIT\00.A_COURS\00.SEMETRE2\BNIL\PROJECT")
    sys.path.append(r"CODE\SPIKING")

    data_directory = r"DATA\data_brut"

    main(data_directory)
