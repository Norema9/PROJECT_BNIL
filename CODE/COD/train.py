import os
import sys
import numpy as np
import ga
import ANN
from trainer import GA_FFNN_TRAINER
import matplotlib.pyplot as plt
import pickle
from dataset import create_dataset
import json

NUM_NEURON_1 = 12 # 8
NUM_NEURON_2 = 6 # 4
NUM_GENERATION = 5000 # 1000
MUTATION_PERCENT = 10
NUM_PARENT_MATING = 4
SOL_PER_POP = 16
SEED = 42
ACTIVATION = "sigmoid"
EVAL_STEP = 10
CHECKPOINT_PATH = None

def main(data_directory):
    data_train_inputs, data_test_inputs, data_train_outputs, data_test_outputs, label_encoder = create_dataset(data_directory)
    input_dim = data_train_inputs.shape[1]
    os.makedirs(os.path.join("MODELS", "accuracy_plot_"+str(NUM_GENERATION)+"__iterations_" + ACTIVATION + "_activation__" +str(MUTATION_PERCENT)+"%_mutation"), exist_ok = True)
    save_dir = os.path.join("MODELS", "accuracy_plot_"+str(NUM_GENERATION)+"__iterations_" + ACTIVATION + "_activation__" +str(MUTATION_PERCENT)+"%_mutation")
    # sys.exit()
    ga_trainer = GA_FFNN_TRAINER(input_dim, NUM_NEURON_1, NUM_NEURON_2, NUM_GENERATION, MUTATION_PERCENT, NUM_PARENT_MATING, SOL_PER_POP, SEED, save_dir, EVAL_STEP, activation = ACTIVATION)
    ga_trainer.initiate_population()

    pop_weights_vector, pop_weights_mat, best_weights, accuracies, loss, duration = ga_trainer.train(data_train_inputs, data_train_outputs, data_test_inputs, data_test_outputs,  MUTATION_PERCENT, checkpoint_path = CHECKPOINT_PATH)
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
    best_weights = pop_weights_mat[0]
    eval_loss, eval_acc, predictions = ANN.predict_outputs(best_weights, data_test_inputs, data_test_outputs, activation = ACTIVATION)
    print("Accuracy of the best solution is : ", eval_acc)
    print("loss of the best solution is : ", eval_loss)


    plt.plot(accuracies, linewidth=2, color = "blue")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.xticks(np.arange(0, NUM_GENERATION+1, 100), fontsize=15)
    plt.yticks(np.arange(0, 101, 5), fontsize=15)

    # Save the plot before showing it
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"), format="png", dpi=300)
    plt.close()


    plt.plot(loss, linewidth = 2, color = "blue")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.xticks(np.arange(0, NUM_GENERATION+1, 100), fontsize=15)
    plt.yticks(np.arange(0, 101, 5), fontsize=15)

    # Save the plot before showing it
    plt.savefig(os.path.join(save_dir, "loss_plot.png"), format="png", dpi=300)
    plt.close()


    f = open(os.path.join(save_dir, "weights.pkl"), "wb")
    pickle.dump(pop_weights_mat, f)
    f.close()

    result = {"loss": loss[-1], "accuracy": accuracies[-1], "duration": duration}
    with open(os.path.join(save_dir, 'results.json'), 'w') as f_result:
        json.dump(result, f_result)


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\02-Documents\00.ETUDES\00.ECOLE_D_INGE\00.CYCLE_ING_FORMATION_INIT\00.3EME_ANNEE_INIT\00.A_COURS\00.SEMETRE2\BNIL\PROJECT")
    sys.path.append(r"CODE\COD")

    data_directory = r"DATA\data_brut"

    main(data_directory)