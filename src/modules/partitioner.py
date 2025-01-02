from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner


class PartitionFactory:
    @staticmethod
    def get_fds(config):
        partitions_number = config.client.count
        dataset_config = config.dataset
        partitioner_name = dataset_config.partitioner.name
        if partitioner_name == "DirichletPartitioner":  # Non IiD
            print("drichlet_________________")
            return FederatedDataset(dataset=dataset_config.name,
                                    subset=dataset_config.subset,
                                    partitioners={"train": DirichletPartitioner(
                                        num_partitions=partitions_number,
                                        partition_by="label",
                                        seed=dataset_config.seed,
                                        alpha=dataset_config.partitioner.alpha,
                                        min_partition_size=0,
                                    )})
        elif partitioner_name == "PathologicalPartitioner":  # Non Iid
            print("pathological__________________")
            return FederatedDataset(dataset=dataset_config.name,
                                    subset=dataset_config.subset,
                                    data_dir=dataset_config.data_dir,
                                    partitioners={"train": PathologicalPartitioner(
                                        num_partitions=partitions_number,
                                        seed=dataset_config.seed,
                                        partition_by="label",
                                        num_classes_per_partition=dataset_config.partitioner.num_classes_per_partition,
                                    )})
        elif partitioner_name == "IiD":  # IiD
            print("iid______________")
            return FederatedDataset(dataset=dataset_config.name,
                                    subset=dataset_config.subset,
                                    partitioners={"train": partitions_number})
        else:
            raise ValueError("Invalid partitioner name")