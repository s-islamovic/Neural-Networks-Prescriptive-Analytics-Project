import random
import numpy as np
from baseline_model import build_model
from tensorflow.keras.callbacks import EarlyStopping

# Genetic Algorithm class for hyperparameter optimization of a neural network
class GeneticAlgorithm:
    def __init__(self, X_train, y_train, population_size=6, generations=4):
        """
        Initialize the Genetic Algorithm.
        
        Parameters:
        - X_train, y_train: training data
        - population_size: number of individuals in each generation
        - generations: number of generations to evolve
        """
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.generations = generations

        # Define possible hyperparameter choices for the neural network
        self.param_choices = {
            "lr": [0.0005, 0.001, 0.005],           # Learning rate options
            "dropout_rate": [0.1, 0.2, 0.3],       # Dropout rates for regularization
            "hidden_units": [16, 32, 64],          # Number of units in hidden layer
            "batch_size": [128, 256],              # Batch size during training
            "epochs": [5]                           # Number of epochs to train
        }

    # Generate a random individual (set of hyperparameters)
    def random_individual(self):
        return {
            "lr": random.choice(self.param_choices["lr"]),
            "dropout_rate": random.choice(self.param_choices["dropout_rate"]),
            "hidden_units": random.choice(self.param_choices["hidden_units"]),
            "batch_size": random.choice(self.param_choices["batch_size"]),
            "epochs": random.choice(self.param_choices["epochs"])
        }

    # Evaluate the fitness of an individual (hyperparameters)
    def fitness(self, params):
        """
        Fitness function: trains the model with given hyperparameters
        and returns the validation accuracy of the last epoch.
        """
        # Build the neural network model with given hyperparameters
        model = build_model(
            input_dim=self.X_train.shape[1],
            lr=params["lr"],
            dropout_rate=params["dropout_rate"],
            hidden_units=params["hidden_units"]
        )

        # Train the model
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_split=0.2,    # Use 20% of training data for validation
            verbose=0,               # Suppress training output
            callbacks=[EarlyStopping(patience=2)]  # Stop early if no improvement
        )

        # Return validation accuracy of the last epoch as fitness score
        return history.history["val_accuracy"][-1]

    # Mutate an individual by randomly changing one hyperparameter
    def mutate(self, individual):
        """
        Randomly changes one hyperparameter in the individual.
        Mutation introduces diversity in the population.
        """
        key = random.choice(list(self.param_choices.keys()))
        individual[key] = random.choice(self.param_choices[key])
        return individual

    # Crossover between two parent individuals to create a child
    def crossover(self, parent1, parent2):
        """
        Combines hyperparameters from two parents.
        Each hyperparameter is randomly selected from either parent.
        """
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    # Main GA loop: evolve the population over generations
    def evolve(self):
        """
        Run the genetic algorithm to find the best hyperparameters.
        Returns the best parameter set found.
        """
        # Step 1: Initialize population with random individuals
        population = [self.random_individual() for _ in range(self.population_size)]

        # Step 2: Evolve over multiple generations
        for gen in range(self.generations):
            print(f"\n--- Generation {gen+1} ---")

            # Evaluate fitness for all individuals in the population
            scores = [(params, self.fitness(params)) for params in population]
            
            # Sort individuals by fitness in descending order (higher is better)
            scores.sort(key=lambda x: x[1], reverse=True)

            # Keep the top 2 individuals as "best"
            best = scores[:2]
            print("Best so far:", best)

            # Step 3: Create next generation
            next_gen = [s[0] for s in best]  # Start with best two

            # Fill the rest of the population using crossover and mutation
            while len(next_gen) < self.population_size:
                parent1, parent2 = random.sample(best, 2)   # Randomly select two parents
                child = self.crossover(parent1[0], parent2[0])  # Create child
                if random.random() < 0.3:                     # 30% chance of mutation
                    child = self.mutate(child)
                next_gen.append(child)

            # Update population for next generation
            population = next_gen

        # Step 4: Return the best hyperparameters found
        best_params = best[0][0]
        print("\nBest Parameters:", best_params)
        return best_params
