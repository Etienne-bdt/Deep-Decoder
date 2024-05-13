clear all

net = dlnetwork;
num_layers=6;
max_iter=5000;
k=32;

img = double(imread('img/astronaut.png'))/255;

noisy_img = img + normrnd(0,0.1,size(img));
%PSNR = psnr(noisy_img, img);

%mask=mask_inpainting(img);
%save mask.mat mask

%imshow(noisy_img);

noisy_img_array = dlarray(noisy_img, "SSCB");

%% Création du NN
Net = [
    imageInputLayer([size(img,1)/2^num_layers size(img,2)/2^num_layers k],"Name","imageinput1","Normalization","none")
    convolution2dLayer([1 1],k,"Name","conv1","Padding","same")
    resize2dLayer("Name","resize-scale1","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])
    reluLayer("Name","relu1")
    batchNormalizationLayer("Name","batchnorm1")];
for i=2:(num_layers)
    tempNet = [
    convolution2dLayer([1 1],k,"Name","conv"+num2str(i),"Padding","same")
    resize2dLayer("Name","resize-scale"+num2str(i),"GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])
    reluLayer("Name","relu"+num2str(i))
    batchNormalizationLayer("Name","batchnorm"+num2str(i))];
    Net = [Net; tempNet];
end
Net = [Net;
    convolution2dLayer([1 1],3, "NumChannels",k, "Name","convlast", "Padding","same")
    sigmoidLayer("Name","sigmoid")
];
net = addLayers(net, Net);

% clean up helper variable
clear tempNet Net;

net = initialize(net);

%% 

options = trainingOptions("adam","Plots","training-progress",ExecutionEnvironment="auto", OutputNetwork="Last-iteration",MaxEpochs=max_iter);

X = dlarray(unifrnd(0,1,size(img,1)/2^num_layers,size(img,2)/2^num_layers,k),"SSCB");


tic;
[trained_net,info]= trainnet(X, img ,net,@loss_inp,options);
toc;

predicted = predict(trained_net, X);

Y = extractdata(predicted);
figure;
imshow(Y);

imwrite(Y, "img\inpainting_test.png");

PSNRY = psnr(double(Y),img);