% setup MatConvNet in MATLAB
run matconvnet/matlab/vl_setupnn

% download a pre−trained CNN from the web
urlwrite ( ...
'http://www.vlfeat.org/sandbox−matconvnet/models/imagenet−vgg−f.mat', ...
'imagenet−vgg−f.mat');
net = load('imagenet−vgg−f.mat');

% obtain and preprocess an image
im 	= imread('peppers.png');
im_ = single(im); % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2));
im_ = im_ − net.normalization.averageImage;
