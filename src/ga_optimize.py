import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

param_space = {
    "lr": [0.0005, 0.001, 0.005],
    "h1": [16, 32, 64],
    "h2": [8, 16, 32],
    "dropout": [0.2, 0.3, 0.4],
    "batch_size": [512, 1024, 2048]
}

def build_model(params):
    model = models.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(params["h1"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["h2"], activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()]
    )

    return model

def evaluate_params(params):
    model = build_model(params)
    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=params["batch_size"],
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history.history["val_auc"][-1]

def random_candidate():
    return {key: random.choice(values) for key, values in param_space.items()}

def mutate(params):
    key = random.choice(list(param_space.keys()))
    params[key] = random.choice(param_space[key])
    return params

def crossover(a, b):
    child = {}
    for key in a.keys():
        child[key] = random.choice([a[key], b[key]])
    return child

def genetic_algorithm(generations=5, population_size=6):
    population = [random_candidate() for _ in range(population_size)]

    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations}")

        scores = [(evaluate_params(ind), ind) for ind in population]
        scores.sort(reverse=True, key=lambda x: x[0])

        population = [x[1] for x in scores[:2]]

        while len(population) < population_size:
            parent1, parent2 = random.sample(scores[:3], 2)
            child = crossover(parent1[1], parent2[1])
            if random.random() < 0.3:
                child = mutate(child)
            population.append(child)

    best_params = scores[0]
    print("\nBest GA Params:", best_params)
    return best_params

if __name__ == "__main__":
    genetic_algorithm()
