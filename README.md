# Benchmark: PyTorch and Flux.jl


### Platform and Version Info

Ubuntu 14.04 with Tesla K40 GPU.


| Tool        | CUDA | Python | PyTorch | Julia | [Flux.jl](https://github.com/FluxML/Flux.jl) | [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) |
| ----------- | ---- | ------ | ------- | ----- | -------------------------------------------- | ------------------------------------------------------ |
| **Version** | 8.0  | 3.5    | 0.4.1   | 1.0.1 | 0.6.7                                        | 0.8.1                                                  |



### Feed Forward Neural Net

mlp

|         | GPU Usage  | Time     | Accuracy |
| ------- | ---------- | -------- | -------- |
| PyTorch | **287 MB** | 89 s     | 0.978    |
| Flux    | 303 MB     | **27 s** | 0.979    |



### Convolutional Neural Net

convnet

|         | GPU Usage  | Time     | Accuracy |
| ------- | ---------- | -------- | -------- |
| PyTorch | **414 MB** | 92 s     | 0.992    |
| Flux    | 1302 MB    | **76 s** | 0.992    |



### Residual Neural Net

resnet

|         | GPU Usage | Time  | Accuracy |
| ------- | --------- | ----- | -------- |
| PyTorch | 769 MB    | 256 s | 0.782    |
| Flux    |           |       |          |

