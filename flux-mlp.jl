using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, @epochs
using CuArrays
include("dataloader.jl")


imgs = MNIST.images()

X = hcat(float.(reshape.(imgs, :))...)
labels = MNIST.labels()
Y = onehotbatch(labels, 0:9)

dataloader = DataLoader(X, Y, batchsize=256, shuffle=true)

m = Chain(
  Dense(28^2, 400, relu),
  Dense(400, 10)) |> gpu

loss(x, y) = logitcrossentropy(m(x |> gpu), y |> gpu)

opt = ADAM(params(m))

@time(@epochs 10 Flux.train!(loss, dataloader, opt))

# Test set accuracy
accuracy(x, y) = mean(onecold(m(x |> gpu)) .== onecold(y |> gpu))

tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

@show accuracy(tX, tY)
