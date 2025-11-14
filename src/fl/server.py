# src/fl/server.py
import flwr as fl
from .strategy import fedavg, weighted_average


def get_strategy(config):
    min_fit = int(config["fl"]["num_clients"] * config["fl"]["fraction_fit"])
    return fl.server.strategy.FedAvg(
        fraction_fit=config["fl"]["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=min_fit,
        min_available_clients=config["fl"]["num_clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": config["fl"]["local_epochs"]
        }
    )