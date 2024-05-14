using Flux, Images, Random, Plots, CUDA

k = 256
max_epochs = 5000
img = load("./img/astronaut.jpg")
#img_noisy = img + 0.25*(rand(eltype(img), size(img)) .- RGB(0.5,0.5,0.5))
#plot(img_noisy)



mask_img = load("./img/mask.jpg")
arr_mask = cat([mask_img for _ in 1:3]..., dims=3) 
arr_mask = reshape(arr_mask, size(arr_mask)..., 1)
arr_mask = convert(Array{Float32}, arr_mask)

arr_img_noisy = permutedims(channelview(img), (2,3,1))
arr_img_noisy = reshape(arr_img_noisy, size(arr_img_noisy)..., 1)
#Convert arr_img_noisy to Float32
arr_img_noisy = convert(Array{Float32}, arr_img_noisy)
arr_img_noisy = arr_img_noisy .* arr_mask |> gpu
arr_mask = arr_mask |> gpu

data = rand(eltype(arr_img_noisy), Int(size(img,1)*(2^-6)), Int(size(img,2)*(2^-6)), k,1)

# Define the model
model = Chain(
    Conv((1,1), k => k;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(k),
    Conv((1,1), k => k;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(k),
    Conv((1,1), k => k;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(k),
    Conv((1,1),k => k;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(k),
    Conv((1,1),k => k;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(k),
    Conv((1,1),k => 3;),
    Upsample(:bilinear, scale=(2,2)),
    relu,
    BatchNorm(3),
    sigmoid
)
model = fmap(cu, model)

opt = Flux.setup(Adam(), model)

train_set = Flux.DataLoader((data, arr_img_noisy)) |> gpu

function loss(model, x, y)
    ŷ = model(x)
    return sum(abs2, y.*arr_mask .- ŷ.*arr_mask)/length(y)
end 
t1 = time();
for epoch in 1:max_epochs
    Flux.train!(loss, model, train_set, opt)
    if epoch % 10 == 0
        println("Epoch: $epoch")
    end
end
println("Time taken for k = $k : ", time()-t1)

model = cpu(model)
im_out = model(data)
im_out = im_out[:,:,:,1]
im_out = permutedims(im_out, (3,1,2))
im_out = colorview(RGB, im_out)

save("deep_decoder.png", im_out)