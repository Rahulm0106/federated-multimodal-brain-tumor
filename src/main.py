# src/main.py
import argparse
import yaml
import flwr as fl
from src.fl.client import FLClient
from src.fl.server import get_strategy


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--modality", choices=["ct", "mri"])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.server:
        strategy = get_strategy(config)
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=config["fl"]["rounds"]),
            strategy=strategy
        )
    else:
        if args.client_id is None or args.modality is None:
            raise ValueError("--client-id and --modality required")
        client = FLClient(args.client_id, args.modality, config)
        fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
    main()