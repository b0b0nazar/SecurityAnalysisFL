#!/bin/bash
 cd ..
# Array to store failed experiments
failed_experiments=()

# Function to run an experiment and track failures
run_experiment() {
  local strategy=$1
  local model=$2
  local dataset=$3
  local partitioner=$4
  local partition_param=$5
  local partition_value=$6
  local num_classes=$7

  # Generate timestamp for directory
  timestamp=$(date +%Y-%m-%d_%H-%M-%S)

  # Define output directory path with timestamp
  output_dir="outputs/${strategy}/${model}/${dataset}/${partitioner}/${partition_value}/${timestamp}"

  echo "Starting experiment with dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"

  # Run the experiment
  python -m src.main hydra.run.dir=$output_dir dataset.subset="$dataset" strategy.name="$strategy" model.name="$model" dataset.partitioner.name="$partitioner" dataset.partitioner.$partition_param="$partition_value" model.num_classes="$num_classes"

  # Check if the experiment failed
  if [ $? -ne 0 ]; then
    echo "Experiment failed: dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"
    # Record the failed experiment
    failed_experiments+=("${dataset}/${partitioner}/${partition_value}")
  fi
}


## Run experiments for pathmnist with PathologicalPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 7 9
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 4 9
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 2 9
#
## Run experiments for pathmnist with DirichletPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "DirichletPartitioner" "alpha" 0.9 9
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "DirichletPartitioner" "alpha" 0.3 9
#run_experiment "FedAvg" "mobilenet_v2" "pathmnist" "DirichletPartitioner" "alpha" 0.1 9
#
## Run experiments for tissuemnist with PathologicalPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 7 8
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 4 8
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 2 8
#
## Run experiments for tissuemnist with DirichletPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "DirichletPartitioner" "alpha" 0.9 8
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "DirichletPartitioner" "alpha" 0.3 8
#run_experiment "FedAvg" "mobilenet_v2" "tissuemnist" "DirichletPartitioner" "alpha" 0.1 8

## Run experiments for bloodmnist with PathologicalPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 2 8
#run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 4 8
#run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 7 8

## Run experiments for bloodmnist with DirichletPartitioner
#run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8
#run_experiment "FedAvg" "mobilenet_v2" "dermamnist" "DirichletPartitioner" "alpha" 0.3 8
#run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8


## Run experiments with different strategies and models

run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8
run_experiment "FedAvg" "resnet101" "bloodmnist" "IiD" "alpha" 0.9 8

run_experiment "FedAvgM" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8
run_experiment "FedAvgM" "resnet101" "bloodmnist" "IiD" "alpha" 0.9 8

run_experiment "FedProx" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8
run_experiment "FedProx" "resnet101" "bloodmnist" "IiD" "alpha" 0.9 8

run_experiment "FedNova" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8
run_experiment "FedNova" "resnet101" "bloodmnist" "IiD" "alpha" 0.9 8


# Final report
echo "Experiments completed!"

# Check if there were any failures
if [ ${#failed_experiments[@]} -ne 0 ]; then
  echo "The following experiments failed:"
  for experiment in "${failed_experiments[@]}"; do
    echo "  - $experiment"
  done
  exit 1  # Exit with error code if any experiments failed
else
  echo "All experiments succeeded!"
  exit 0  # Exit with success code if everything went fine
fi