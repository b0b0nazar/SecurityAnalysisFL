import flwr as fl
from src.modules.server import fit_config, weighted_average, ServerFactory, trimmed_mean_aggregate, krum_aggregate, \
    median_aggregate


class StrategyFactory:
    @staticmethod
    def get_strategy(centralized_testset, config):

        if config.strategy.name == "FedAvg":
            return fl.server.strategy.FedAvg(
                # Sample 10% of available clients for training
                fraction_fit=config.strategy.fraction_train_clients,
                fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
                min_available_clients=3,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=median_aggregate,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function

            )
        elif config.strategy.name == "FedAvgM":
            return fl.server.strategy.FedAvgM(
                # Sample 10% of available clients for training
                fraction_fit=config.strategy.fraction_train_clients,
                fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
                min_available_clients=3,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function

            )
        elif config.strategy.name == "FedProx":
            return fl.server.strategy.FedProx(
                proximal_mu=1,
                # Sample 10% of available clients for training
                fraction_fit=config.strategy.fraction_train_clients,
                fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
                min_available_clients=3,
                on_fit_config_fn=fit_config,
                # evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )
        elif config.strategy.name == "FedNova":
            return fl.server.strategy.FedAvg(
                # Sample 10% of available clients for training
                fraction_fit=config.strategy.fraction_train_clients,
                fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
                min_available_clients=3,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )

        else:
            raise ValueError(f"Invalid strategy {config.strategy.name}")
