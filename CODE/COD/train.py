import os
import sys
import numpy as np
import ga
import ANN
from trainer import GA_FFNN_TRAINER
import matplotlib.pyplot
import pickle
from dataset import create_dataset

NUM_NEURON_1 = 150
NUM_NEURON_2 = 60
NUM_NEURON_3 = 10
NUM_GENERATION = 20
MUTATION_PERCENT = 10
NUM_PARENT_MATING = 4
SOL_PER_POP = 8

def main(data_directory):
    data_train_inputs, data_train_outputs, data_test_inputs, data_test_outputs, label_encoder = create_dataset(data_directory)
    input_dim = data_train_inputs.shape[1]

    # sys.exit()
    ga_trainer = GA_FFNN_TRAINER(input_dim, NUM_NEURON_1, NUM_NEURON_2, NUM_NEURON_3, NUM_GENERATION, MUTATION_PERCENT, NUM_PARENT_MATING, SOL_PER_POP)
    ga_trainer.initiate_population()

    pop_weights_vector, pop_weights_mat, best_weights, accuracies, loss = ga_trainer.train(data_train_inputs, data_train_outputs, MUTATION_PERCENT)
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
    best_weights = pop_weights_mat [0, :]
    eval_loss, eval_acc, predictions = ANN.predict_outputs(best_weights, data_test_inputs, data_test_outputs, activation = "sigmoid")
    print("Accuracy of the best solution is : ", eval_acc)
    print("loss of the best solution is : ", eval_loss)

    matplotlib.pyplot.plot(accuracies, linewidth = 5, color="black")
    matplotlib.pyplot.xlabel("Iteration", fontsize=20)
    matplotlib.pyplot.ylabel("Fitness", fontsize=20)
    matplotlib.pyplot.xticks(np.arange(0, NUM_GENERATION+1, 100), fontsize=15)
    matplotlib.pyplot.yticks(np.arange(0, 101, 5), fontsize=15)
    matplotlib.pyplot.show()

    f = open(os.path.join("MODEL", "weights_"+str(NUM_GENERATION)+"_iterations_"+str(MUTATION_PERCENT)+"%_mutation.pkl"), "wb")
    pickle.dump(pop_weights_mat, f)
    f.close()


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\02-Documents\00.ETUDES\00.ECOLE_D_INGE\00.CYCLE_ING_FORMATION_INIT\00.3EME_ANNEE_INIT\00.A_COURS\00.SEMETRE2\BNIL\PROJECT")
    sys.path.append(r"CODE\COD")

    data_directory = r"DATA\data_brut"

    main(data_directory)