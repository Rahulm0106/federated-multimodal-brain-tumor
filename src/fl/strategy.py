from typing import List
import numpy as np


def fedavg(weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    avg_weights = []
    for weights_list_tuple in zip(*weights):
        layer_weights = np.array(weights_list_tuple)
        avg_weights.append(layer_weights.mean(axis=0))
    return avg_weights


def weighted_average(metrics: List[tuple]) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}