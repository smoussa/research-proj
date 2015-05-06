% load the CNN
net = load('imagenet-vgg-s.mat');

% obtain and preprocess an image
im 	= imread('peppers.png');
im_ = single(im);
im_ = imresize(im_, net.normalization.imageSize(1:2));
im_ = im_ - net.normalization.averageImage;
res = vl_simplenn(net,im_);

% show the classification result
scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1); clf; imagesc(im);
title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));
% savefig('figure.fig');