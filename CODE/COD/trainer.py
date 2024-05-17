import numpy
import ga
import ANN

class GA_FFNN_TRAINER:
    def __init__(self, 
                 input_dim, 
                 num_neurons_1, 
                 num_neurons_2, 
                 num_neurons_3, 
                 num_generations, 
                 mutation_percent,
                 num_parents_mating, 
                 sol_per_pop,) -> None:
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
        self.HL3_neurons = num_neurons_3
        self.input_dim = input_dim

    def initiate_population(self):
        #Creating the initial population.
        for curr_sol in numpy.arange(0, self.sol_per_pop):
            input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.input_dim, self.HL1_neurons))
            HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.HL1_neurons, self.HL2_neurons))
            HL2_HL3_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.HL2_neurons, self.HL3_neurons))
            output_neurons = 1
            HL3_output_weights = numpy.random.uniform(low=-0.1, high=0.1, 
                                                    size=(self.HL3_neurons, output_neurons))

            print(input_HL1_weights.shape)
            print(HL1_HL2_weights.shape)
            print(HL2_HL3_weights.shape)
            print(HL3_output_weights.shape)

            self.initial_pop_weights.append([input_HL1_weights, 
                                                        HL1_HL2_weights, 
                                                        HL2_HL3_weights, 
                                                        HL3_output_weights])
        self.pop_weights_mat = self.initial_pop_weights
        self.pop_weights_vector = ga.mat_to_vector(self.pop_weights_mat)

        self.best_outputs = []
        self.accuracies = numpy.empty(shape=(self.num_generations))
        self.loss = numpy.empty(shape=(self.num_generations))


    def train(self, data_inputs, data_outputs, mutation_percent):
        for generation in range(self.num_generations):

            # converting the solutions from being vectors to matrices.
            self.pop_weights_mat = ga.vector_to_mat(self.pop_weights_vector, 
                                            self.pop_weights_mat)

            # Measuring the fitness of each chromosome in the population.
            fitness_loss, fitness_acc = ANN.fitness(self.pop_weights_mat, 
                                data_inputs, 
                                data_outputs, 
                                activation="sigmoid")
            self.accuracies[generation] = fitness_acc[0]
            self.loss[generation] = fitness_loss[0]
            

            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(self.pop_weights_vector, 
                                            fitness_loss.copy(), 
                                            self.num_parents_mating)
            # print("Parents")
            # print(parents)

            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
                                            offspring_size=(self.pop_weights_vector.shape[0]-parents.shape[0], self.pop_weights_vector.shape[1]))
            # print("Crossover")
            # print(offspring_crossover)

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, 
                                            mutation_percent=mutation_percent)
            # print("Mutation")
            # print(offspring_mutation)

            # Creating the new population based on the parents and offspring.
            self.pop_weights_vector[0:parents.shape[0], :] = parents
            self.pop_weights_vector[parents.shape[0]:, :] = offspring_mutation
            best_weights = self.pop_weights_mat [0, :]

            progress = "="*int((generation/self.num_generations)*10) + ">" + ".."*int(10 - (generation/self.num_generations)*10)
            
            print(f"{progress}| Generation : {generation}| Accuracy : {fitness_acc}| Loss : {fitness_loss}")

        return self.pop_weights_vector, self.pop_weights_mat, best_weights, self.accuracies, self.loss
