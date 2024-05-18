import numpy
import ga
import ANN
import time
import pickle
import os
import pandas as pd

class GA_FFNN_TRAINER:
    def __init__(self, 
                 input_dim, 
                 num_neurons_1, 
                 num_neurons_2, 
                #  num_neurons_3, 
                 num_generations, 
                 mutation_percent,
                 num_parents_mating, 
                 sol_per_pop,
                 seed,
                 save_dir,
                 eval_step,
                 activation = "sigmoid") -> None:
        """
        Genetic algorithm parameters:
            Mating Pool Size (Number of Parents)
            Population Size
            Number of Generations
            Mutation Percent
        """
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_percent = mutation_percent
        self.initial_pop_weights = []
        self.HL1_neurons = num_neurons_1
        self.HL2_neurons = num_neurons_2
        # self.HL3_neurons = num_neurons_3
        self.input_dim = input_dim
        self.activation = activation
        self.seed = seed
        self.eval_step = eval_step
        self.save_dir = save_dir

    def initiate_population(self):
        #Creating the initial population.
        numpy.random.seed(self.seed)
        for curr_sol in numpy.arange(0, self.sol_per_pop):
            layers = {}
            input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.input_dim, self.HL1_neurons))
            HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.HL1_neurons, self.HL2_neurons))
            # HL2_HL3_weights = numpy.random.uniform(low=-0.1, high=0.1, 
            #                                         size=(self.HL2_neurons, self.HL3_neurons))
            # output_neurons = 1
            # HL3_output_weights = numpy.random.uniform(low=-0.1, high=0.1, 
            #                                         size=(self.HL3_neurons, output_neurons))
            
            output_neurons = 1
            HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.HL2_neurons, output_neurons))

            layers["input_layer"] = input_HL1_weights
            layers["hidden_layer_1"] = HL1_HL2_weights
            # layers["hidden_layer_2"] = HL2_HL3_weights
            # layers["output_layer"] = HL3_output_weights
            layers["output_layer"] = HL2_output_weights

            self.initial_pop_weights.append(layers)

        self.pop_weights_mat = self.initial_pop_weights
        self.pop_weights_vector = ga.mat_to_vector(self.pop_weights_mat)

        self.best_outputs = []
        self.accuracies = numpy.empty(shape=(self.num_generations))
        self.loss = numpy.empty(shape=(self.num_generations))
        self.best_acc = 0
        
        self.val = []
        self.trai = []


    def save_checkpoint(self, generation, best_weights):
        checkpoint = {
            'generation': generation,
            'pop_weights_vector': self.pop_weights_vector,
            'pop_weights_mat': self.pop_weights_mat,
            'best_weights': best_weights,
            'accuracies': self.accuracies,
            'loss': self.loss,
            'best_acc': self.best_acc,
            'val': self.val,
            'trai': self.trai,
            'seed': self.seed,
            'elapsed_time': time.time() - self.start_time
        }
        with open(os.path.join(self.save_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.pop_weights_vector = checkpoint['pop_weights_vector']
        self.pop_weights_mat = checkpoint['pop_weights_mat']
        self.best_weights = checkpoint['best_weights']
        self.accuracies = checkpoint['accuracies']
        self.loss = checkpoint['loss']
        self.best_acc = checkpoint['best_acc']
        self.val = checkpoint['val']
        self.trai = checkpoint['trai']
        self.seed = checkpoint['seed']
        self.start_time = time.time() - checkpoint['elapsed_time']
        return checkpoint['generation']


    def train(self, data_train_inputs, data_train_outputs, data_test_inputs, data_test_outputs, mutation_percent, checkpoint_path=None):
        if checkpoint_path:
            start_generation = self.load_checkpoint(checkpoint_path)
        else:
            start_generation = 0
            self.start_time = time.time()

        for generation in range(start_generation, self.num_generations):
            # converting the solutions from being vectors to matrices.
            self.pop_weights_mat = ga.vector_to_mat(self.pop_weights_vector, 
                                            self.pop_weights_mat)

            # Measuring the fitness of each chromosome in the population.
            fitness_loss, fitness_acc = ANN.fitness(self.pop_weights_mat, 
                                data_train_inputs, 
                                data_train_outputs, 
                                activation=self.activation)
            self.accuracies[generation] = fitness_acc[0]
            self.loss[generation] = fitness_loss[0]
            

            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(self.pop_weights_vector, 
                                            fitness_acc.copy(), 
                                            self.num_parents_mating)

            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
                                            offspring_size=(self.pop_weights_vector.shape[0]-parents.shape[0], self.pop_weights_vector.shape[1]))

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, 
                                            mutation_percent=mutation_percent, seed = self.seed)

            # Creating the new population based on the parents and offspring.
            self.pop_weights_vector[0:parents.shape[0], :] = parents
            self.pop_weights_vector[parents.shape[0]:, :] = offspring_mutation
            best_weights = self.pop_weights_mat[0]

            elapsed_time = time.time() - self.start_time
            progress = "="*int((generation/self.num_generations)*50) + ">" + "*"*int(50 - (generation/self.num_generations)*50)
            percent = int((generation/self.num_generations)*100)

            progress_bar = f"{percent}%| {progress}| Generation : {generation}| Accuracy : {numpy.mean(fitness_acc):.3f}| Loss : {numpy.mean(fitness_loss):.3f}| duration : {elapsed_time:.4f}"

            if (generation + 1) % self.eval_step == 0:
                eval_loss, eval_acc, predictions = ANN.predict_outputs(best_weights, data_test_inputs, data_test_outputs, activation = self.activation)
                self.val.append({"generation": generation, "val_accuracy": eval_acc, "val_loss": eval_loss, "train_accuracy": numpy.mean(fitness_acc), "train_loss": numpy.mean(fitness_loss)})
                if eval_acc > self.best_acc:
                        f = open(os.path.join(self.save_dir, "best_weights.pkl"), "wb")
                        pickle.dump(best_weights, f)
                        f.close()
                self.save_checkpoint(generation, best_weights)

            if not self.val == []:
                 acc = self.val[-1]["val_accuracy"]
                 los = self.val[-1]["val_loss"]
                 progress_bar += f"| eval_acc : {acc:.3f}| eval_loss : {los:.3f}"
            
            print(progress_bar)
        eval_df = pd.DataFrame(self.val)
        eval_df.to_csv(os.path.join(self.save_dir, "train_eval_results_progress.csv"))
        return self.pop_weights_vector, self.pop_weights_mat, best_weights, self.accuracies, self.loss, elapsed_time
