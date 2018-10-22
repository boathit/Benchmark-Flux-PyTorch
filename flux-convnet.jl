using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, @epochs
using CuArrays

include("dataloader.jl")

function fastcat(xs; dims)
    length(xs) <= 200 && return cat(xs..., dims=dims)
    m = div(length(xs), 2)
    cat(fastcat(xs[1:m], dims=dims), fastcat(xs[m+1:end], dims=dims), dims=dims)
end

trainimgs = MNIST.images()
trainlabels = MNIST.labels()
testimgs = MNIST.images(:test)
testlabels = MNIST.labels(:test)

X = fastcat(float.(trainimgs), dims=4)
Y = onehotbatch(trainlabels, 0:9)
tX = fastcat(float.(testimgs), dims=4)
tY = onehotbatch(testlabels, 0:9)

trainloader = DataLoader(X, Y, batchsize=256, shuffle=true)

m = Chain(
  Conv((5,5), 1=>20),
  x -> maxpool(x, (2,2)),
  x -> relu.(x),
  Conv((5,5), 20=>50),
  x -> maxpool(x, (2,2)),
  x -> relu.(x),
  x -> reshape(x, :, size(x, 4)),
  Dense(800, 500),
  x -> relu.(x),
  Dense(500, 10)) |> gpu

# (x, y) = first(trainloader)
# m(x)

loss(x, y) = logitcrossentropy(m(x |> gpu), y |> gpu)

opt = ADAM(params(m))

@time(@epochs 10 Flux.train!(loss, trainloader, opt))

accuracy(x, y) = mean(onecold(m(x |> gpu)) .== onecold(y |> gpu))
@show accuracy(tX, tY)
