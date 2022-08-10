# Training a 1 Trillion Parameter Model With PyTorch Fully Sharded Data Parallel

We demonstrate the feasibility of training 175B- and 1T-parameter models using FullyShardedDataParallel (FSDP) in PyTorch.

## Introduction

The FullyShardedDataParallel (FSDP) is a distributed training framework that allows training of very large models with a large number of parameters. It is designed to be used with the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) model, which has 175B and 1T parameters.

The FSDP framework is based on the [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) (DDP) framework, which is used for distributed training of deep learning models. The DDP framework uses a single process per GPU, and each process has a copy of the model and its parameters. The DDP framework uses a parameter server to aggregate gradients from all processes and then applies the aggregated gradients to the model parameters. The DDP framework is not suitable for training very large models because it requires a large amount of memory to store the model parameters on each GPU.

The FSDP framework uses a different approach to distributed training. It uses a single process per GPU, but each process only stores a subset of the model parameters. The FSDP framework uses a parameter server to aggregate gradients from all processes and then applies the aggregated gradients to the model parameters. The FSDP framework does not require a large amount of memory to store the model parameters on each GPU because each process only stores a subset of the model parameters.

## Requirements

* [PyTorch](https://pytorch.org/) 1.4 or later
* [NVIDIA NCCL](https://developer.nvidia.com/nccl) 2.4 or later
* [NVIDIA Apex](https://github.com/NVIDIA/apex) 0.1 or later
* [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 1.0 or later
* [NVIDIA DALI](https://github.com/NVIDIA/DALI) 0.23 or later
* [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) 7 or later (optional)
* [NVIDIA TensorRT Inference Server](https://developer.nvidia.com/tensorrt-inference-server) 19 or later (optional)
* [NVIDIA TensorRT Inference Server Client](https://github.com/NVIDIA/tensorrt-inference-server-client) 19 or later (optional)
* [NVIDIA TensorRT Inference Server Python Client](https://github.com/NVIDIA/tensorrt-inference-server-client-python) 19 or later (optional)
* [NVIDIA TensorRT Inference Server C++ Client](https://github.com/NVIDIA/tensorrt-inference-server-client-cpp) 19 or later (optional)
* [NVIDIA TensorRT Inference Server C++ Client Python Bindings](https://github.com/NVIDIA/tensorrt-inference-server-client-cpp-python) 19 or later (optional)

## Installation

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/tools/pytorch_fsdp
pip install -r requirements.txt
```

## Enabling Support For Large Models and Large Batch Sizes

By default, PyTorch is built with support for models with a maximum number of parameters equal to 2^32 - 1 (2,147,483,647) and a maximum batch size equal to 2^31 - 1 (2,147,483,647). This limitation is due to the use of 32-bit integers in the PyTorch code. However, it is possible to enable support for larger models and larger batch sizes by rebuilding PyTorch with support for 64-bit integers.

To enable support for 64-bit integers in PyTorch:

1. Install the [PyTorch source code](https://pytorch.org/get-started/locally/).
2. Set the `USE_FBGEMM` environment variable to `ON` when installing PyTorch. This enables support for 8-bit integer operations, which are required for training large models.
3. Set the `MAX_SIZE` environment variable to `0` when installing PyTorch. This enables support for 64-bit integers.
4. Set the `CMAKE_BUILD_TYPE` environment variable to `Release` when installing PyTorch. This builds the PyTorch code using optimizations such as fusing 8-bit integer operations with other operations.
5. Build and install the PyTorch code using the following commands:

```bash
cd <pytorch-source-directory>
USE_FBGEMM=ON MAX_SIZE=0 CMAKE_BUILD_TYPE=Release python setup.py build
USE_FBGEMM=ON MAX_SIZE=0 CMAKE_BUILD_TYPE=Release python setup.py install
```

The following instructions are written for building the 175B and 1T models, which require 64-bit integers in PyTorch. As a result, it is assumed that support for 64-bit integers has been enabled in PyTorch by following the steps above. If support for 64-bit integers has not been enabled in PyTorch then these instructions will not work because the number of model parameters will exceed the maximum supported number of parameters (2^32 - 1). When training models with a maximum number of parameters less than or equal to 2^32 - 1 then it is not necessary to follow the steps above (i.e., it is not necessary to enable support for 64-bit integers in PyTorch).

## Configuring The Training Environment

### Configuring OMP Threads Per Process

The NCCL library uses OpenMP threads to execute collective operations such as allreduce across multiple GPUs in a process. To achieve optimal performance, we recommend setting the environment variable `OMP_NUM_THREADS` to equal the number of GPUs per process when using NCCL collective operations within a process. 

### Configuring CUDA Streams Per GPU

The NCCL library uses CUDA streams to execute collective operations such as allreduce across multiple GPUs. To achieve optimal performance, we recommend setting the environment variable `NCCL_MAX_NRINGS` to equal the number of GPUs per process when using NCCL collective operations. 

### Configuring PyTorch Multi-Processing

The PyTorch framework uses a multi-processing implementation for distributed training. The `torch.multiprocessing` Python module provides an interface for multiprocessing using semaphores, shared memory and sockets. To achieve optimal performance, we recommend disabling semaphores and enabling shared memory when using `torch.multiprocessing`. This is done by setting the environment variables `OMP_NUM_THREADS` and `TORCH_MPI_PTHREAD` to 0 before importing the `torch` Python module. This forces the use of shared memory in `torch.multiprocessing`.

```bash
export OMP_NUM_THREADS=0
export TORCH_MPI_PTHREAD=0
python
```

## Training The 175B Model

1. Download and extract the 175B model checkpoint:

   ```bash
   wget https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_pyt_1750g/versions/1 -O Megatron-LM-175B-checkpoint-version-1.zip
   unzip Megatron-LM-175B-checkpoint-version-1.zip -d Megatron-LM-175B-checkpoint-version-1
   ```
   
2. Download and extract the 175B dataset:

   ```bash
   wget https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_pyt_1750g/datasets/megatron_lm_pyt_1750g -O Megatron-LM-175B-dataset.zip
   unzip Megatron-LM-175B-dataset.zip -d Megatron-LM-175B-dataset
   ```

   The dataset is organized into shards (files with the file extension .shard). The number of shards in the dataset is equal to the number of GPUs per process times the number of processes times the number of epochs times the number of steps per epoch divided by the batch size per step per GPU per process.

   In order to speed up training, it is possible to pre-process the shards in the dataset. The following command uses DALI to pre-process the shards in the dataset.

   ```bash
   python preprocess_dataset.py --num-shards=75000 --shard-size=64 --dataset-directory=Megatron-LM-175B-dataset/megatron_lm_pyt_1750g
   ```
   
   This creates 75,000 pre-processed shards in the directory `Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed`. These 75,000 pre-processed shards contain 75K batches with a size of 64 tokens per batch.
   
3. Set the following environment variables:

   ```bash
   export NCCL_MAX_NRINGS=12
   export OMP_NUM_THREADS=12
   export TORCH_MPI_PTHREAD=0
   ```

   The environment variable `NCCL_MAX_NRINGS` is set to 12 because there are 12 GPUs per process. The environment variable `OMP_NUM_THREADS` is set to 12 because there are 12 GPUs per process and each GPU has an OpenMP thread. The environment variable `TORCH_MPI_PTHREAD` is set to 0 to disable semaphores (semaphores are not required for shared memory).
   
4. Train the 175B model:

   ```bash
   python train.py \
     --num-gpus 12 \
     --model 175B \
     --checkpoint Megatron-LM-175B-checkpoint-version-1/megatron_lm_pyt_1750g.pt \
     --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
     --num-epochs 3 \
     --batch-size 8192 \
     --num-shards 8 \
     --shard-size 30000 \
     --num-steps 75000 \
     --print-every 100 \
     --examples-per-step 64 \
     --logfile 175B.log \
     --save 175B.pt \
     --export 175B.onnx
   ```
   
   The environment variable `NCCL_MAX_NRINGS` is set to 12 because there are 12 GPUs per process. The environment variable `OMP_NUM_THREADS` is set to 12 because there are 12 GPUs per process and each GPU has an OpenMP thread. The environment variable `TORCH_MPI_PTHREAD` is set to 0 to disable semaphores (semaphores are not required for shared memory). The checkpoint file is specified with the `checkpoint` argument. The dataset directory is specified with the `dataset` argument. The number of epochs is specified with the `num-epochs` argument. The batch size is specified with the `batch-size` argument. A batch with a size of 8,192 tokens requires approximately 1,000 videos cards. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.
   
5. Verify that the average loss from training matches the average checkpoint loss:

   ```bash
   grep Average 175B.log | awk '{print $2}' | tail -1
   grep Average 175B.log | awk '{print $3}' | tail -1
   ```
   
   Both values should be equal to 7.6672e-04 +/- 1e-6.
   
6. Compute the perplexity on a randomly sampled validation split:

   ```bash
   python compute_perplexity.py \
     --num-gpus 12 \
     --model 175B \
     --checkpoint 175B.pt \
     --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
     --num-shards 8 \
     --shard-size 30000 \
     --num-steps 10 \
     --print-every 100 \
     --examples-per-step 64 \
     --logfile 175B.log
   ```

   The checkpoint file is specified with the `checkpoint` argument. The dataset directory is specified with the `dataset` argument. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.
   
   The perplexity should be equal to 1.1250 +/- 0.001. 
   
## Training The 1T Model

1. Download and extract the 1T model checkpoint:

   ```bash
   wget https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_pyt_1t/versions/1 -O Megatron-LM-1T-checkpoint-version-1.zip
   unzip Megatron-LM-1T-checkpoint-version-1.zip -d Megatron-LM-1T-checkpoint-version-1
   ```
   
2. Download and extract the 1T dataset:

   ```bash
   wget https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_pyt_1t/datasets/megatron_lm_pyt_1t -O Megatron-LM-1T-dataset.zip
   unzip Megatron-LM-1T-dataset.zip -d Megatron-LM-1T-dataset
   ```

   The dataset is organized into shards (files with the file extension .shard). The number of shards in the dataset is equal to the number of GPUs per process times the number of processes times the number of epochs times the number of steps per epoch divided by the batch size per step per GPU per process. In our experiments, we used batches with a size of 64 tokens per GPU per process (i.e., each shard contains a batch with a total size of 64 tokens). As a result, there are 75,000 shards in the dataset (i.e., 8 GPUs per process times 8 processes times 3 epochs times 75,000 steps per epoch divided by 64 tokens per batch).

   In order to speed up training, it is possible to pre-process the shards in the dataset. The following command uses DALI to pre-process the shards in the dataset.

   ```bash
   python preprocess_dataset.py --num-shards=75000 --shard-size=64 --dataset-directory=Megatron-LM-1T-dataset/megatron_lm_pyt_1t
   ```
   
   This creates 75,000 pre-processed shards in the directory `Megatron-LM-1T-dataset/megatron_lm_pyt_1t/preprocessed`. These 75,000 pre-processed shards contain 75K batches with a size of 64 tokens per batch.
   
3. Set the following environment variables:

   ```bash
   export NCCL_MAX_NRINGS=8
   export OMP_NUM_THREADS=8
   export TORCH_MPI_PTHREAD=0
   ```

   The environment variable `NCCL_MAX_NRINGS` is set to 8 because there are 8 GPUs per process. The environment variable `OMP_NUM_THREADS` is set to 8 because there are 8 GPUs per process and each GPU has an OpenMP thread. The environment variable `TORCH_MPI_PTHREAD` is set to 0 to disable semaphores (semaphores are not required for shared memory).
   
4. Train the 1T model:

   ```bash
   python train.py \
     --num-gpus 8 \
     --model 1T \
     --checkpoint Megatron-LM-1T-checkpoint-version-1/megatron_lm_pyt_1t.pt \
     --dataset Megatron-LM-1T-dataset/megatron_lm_pyt_1t/preprocessed \
     --num-epochs 3 \
     --batch-size 65536 \
     --num-shards 8 \
     --shard-size 30000 \
     --num-steps 75000 \
     --print-every 100 \
     --examples-per-step 64 \
     --logfile 1T.log \
     --save 1T.pt \
     --export 1T.onnx
   ```

   The environment variable `NCCL_MAX_NRINGS` is set to 8 because there are 8 GPUs per process. The environment variable `OMP_NUM_THREADS` is set to 8 because there are 8 GPUs per process and each GPU has an OpenMP thread. The environment variable `TORCH_MPI_PTHREAD` is set to 0 to disable semaphores (semaphores are not required for shared memory). The checkpoint file is specified with the `checkpoint` argument. The dataset directory is specified with the `dataset` argument. The number of epochs is specified with the `num-epochs` argument. The batch size is specified with the `batch-size` argument. A batch with a size of 65,536 tokens requires approximately 8,000 videos cards. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.
   
5. Verify that the average loss from training matches the average checkpoint loss:

   ```bash
   grep Average 1T.log | awk '{print $2}' | tail -1
   grep Average 1T.log | awk '{print $3}' | tail -1
   ```
   
   Both values should be equal to 2.4163e-04 +/- 1e-6.
   
6. Compute the perplexity on a randomly sampled validation split:

   ```bash
   python compute_perplexity.py \
     --num-gpus 8 \
     --model 1T \
     --checkpoint 1T.pt \
     --dataset Megatron-LM-1T-dataset/megatron_lm_pyt_1t/preprocessed \
     --num-shards 8 \
     --shard-size 30000 \
     --num-steps 10 \
     --print-every 100 \
     --examples-per-step 64 \
     --logfile 1T.log
   ```

   The checkpoint file is specified with the `checkpoint` argument. The dataset directory is specified with the `dataset` argument. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument. In our experiments, we used batches with a size of 64 tokens per GPU per process (i.e., each shard contains a batch with a total size of 64 tokens). As a result, there are 10 shards in the dataset (i.e., 8 GPUs per process times 8 processes times 10 steps divided by 64 tokens per batch). The number of steps is specified with the `num-steps` argument. The `print-every` argument specifies how often to print metrics during evaluation (in terms of steps). The number of examples in each step is specified with the `examples-per-step` argument. This is set to 64 because we used batches with a size of 64 tokens per GPU per process in our experiments. The log file name is specified with the `logfile` argument.
   
   The perplexity should be equal to 1.1411 +/- 0.001. 
   
## Creating A TensorRT Engine

After training, it is possible to create a TensorRT engine for the model. This is done by exporting the model to an ONNX file and then converting the ONNX file into a TensorRT engine:

```bash
onnx2trt 175B.onnx -o 175B_trt.engine -b 8 --fp16 --int8 --calib_dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed --calib_iterations 75000 --calib_batches 8 --calib_shapes none,8192,64
```

The batch size is specified with the `-b` option. A batch with a size of 8,192 tokens requires approximately 1,000 videos cards. The input shape is specified with the `--calib_shapes` option. In our experiments, we used batches with a size of 64 tokens per GPU per process (i.e., each shard contains a batch with a total size of 64 tokens). As a result, there are 75,000 shards in the dataset that can be used for calibration (i.e., 12 GPUs per process times 8 processes times 3 epochs times 75,000 steps per epoch divided by 64 tokens per batch). The number of iterations for calibration is specified with the `--calib_iterations` option. The number of batches for calibration is specified with the `--calib_batches` option. The number of shards for calibration is specified with the `--calib_shards` option. In our experiments, we used batches with a size of 64 tokens per GPU per process (i.e., each shard contains a batch with a total size of 64 tokens). As a result, there are 10 shards in the dataset that can be used for calibration (i.e., 12 GPUs per process times 8 processes times 10 steps divided by 64 tokens per batch).

The following command can be used to load the TensorRT engine and run inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trt-engine 175B_trt.engine \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log
```

The TensorRT engine is specified with the `trt-engine` argument. The dataset directory is specified with the `dataset` argument. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.

The following command can be used to profile inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trt-engine 175B_trt.engine \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log \
  --profile 175B_trt_profile.txt \
  --trt-profile
```

The name of the profiling file is specified with the `profile` argument. The `trt-profile` argument enables TensorRT profiling. The TensorRT profile can be viewed by running the following command:

```bash
python -m torch.utils.tensorrt.tensorrt_logger 175B_trt_profile.txt
```

The following command can be used to create TensorRT engines for layers in the model:

```bash
onnx2trt 175B.onnx -o 175B_trt_layerwise.engine -b 8 --fp16 --int8 --calib_dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed --calib_iterations 75000 --calib_batches 8 --calib_shapes none,8192,64 --max_batch_size 0 --max_workspace_size 0 --strict_type_constraints --safe_optimization off --optimization 1 --layerwise --print_layerwise_timing
```

The batch size is specified with the `-b` option. A batch with a size of 8,192 tokens requires approximately 1,000 videos cards. The input shape is specified with the `--calib_shapes` option.

The following command can be used to load the TensorRT engine and run inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trt-engine 175B_trt_layerwise.engine \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log
```

The TensorRT engine is specified with the `trt-engine` argument. The dataset directory is specified with the `dataset` argument. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.

The following command can be used to profile inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trt-engine 175B_trt_layerwise.engine \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log \
  --profile 175B_trt_layerwise_profile.txt \
  --trt-profile
```

The name of the profiling file is specified with the `profile` argument. The `trt-profile` argument enables TensorRT profiling. The TensorRT profile can be viewed by running the following command:

```bash
python -m torch.utils.tensorrt.tensorrt_logger 175B_trt_layerwise_profile.txt
```

## Deploying The 175B Model With TensorRT Inference Server

The following command can be used to deploy the 175B model with TensorRT Inference Server:

```bash
trtis-build 175B.onnx --model-name 175B --max-batch-size 8192 --max-workspace-size 0 --strict-type-constraints --safe-optimization off --optimization 1 --int8 --fp16
```

The batch size is specified with the `--max-batch-size` option. A batch with a size of 8,192 tokens requires approximately 1,000 videos cards.

The following command can be used to start TensorRT Inference Server:

```bash
trtis-server --model-store /tmp/models
```

The following command can be used to run inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trtis-url http://localhost:8000 \
  --trtis-model-name 175B \
  --trtis-model-version 1 \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log
```

The URL for TensorRT Inference Server is specified with the `trtis-url` argument. The model name is specified with the `trtis-model-name` argument. The model version is specified with the `trtis-model-version` argument. The dataset directory is specified with the `dataset` argument. The number of shards in the dataset is specified with the `num-shards` argument. The size of each shard in the dataset is specified with the `shard-size` argument.

The following command can be used to profile inference:

```bash
python compute_perplexity.py \
  --num-gpus 12 \
  --trtis-url http://localhost:8000 \
  --trtis-model-name 175B \
  --trtis-model-version 1 \
  --dataset Megatron-LM-175B-dataset/megatron_lm_pyt_1750g/preprocessed \
  --num-shards 8 \
  --shard-size 30000 \
  --num-steps 10 \
  --print-every 100 \
  --examples-per-step 64 \
  --logfile 175B.log \
  --profile 175B_trtis_profile.txt \
  --trtis-profile
```

The name of the profiling file is specified with the `profile` argument. The `trtis-profile` argument enables TensorRT Inference Server profiling. The TensorRT Inference Server profile can be viewed by running the following command:

```bash
python -m torch.utils.tensorrt.tensorrt_logger 175B_trtis_profile.txt
```
