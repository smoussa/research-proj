% % setup MatConvNet in MATLAB
% run matconvnet/matlab/vl_setupnn

% % download a pre−trained CNN from the web
% % urlwrite ( ...
% % 'http://www.vlfeat.org/sandbox−matconvnet/models/imagenet−vgg−f.mat', ...
% % 'imagenet−vgg−f.mat');
% net = load('imagenet-vgg-s.mat');

% % obtain and preprocess an image
% im 	= imread('peppers.png');
% im_ = single(im); % note: 255 range
% im_ = imresize(im_, net.normalization.imageSize(1:2));
% im_ = im_ - net.normalization.averageImage;


function irp_setup()

% -------------------------------------------------------------------------
%	Instructions
% -------------------------------------------------------------------------
%{
	
	() Create data/ directory
	
	()	Download ImageNet data from:
			http://www.image-net.org/challenges/LSVRC
			http://www.image-net.org/download-images (Registration needed)

	()	The ILSVRC data ships in several TAR archives that can be obtained
		from the ILSVRC challenge website. You will need:
		   ILSVRC2012_img_train.tar
		   ILSVRC2012_img_val.tar
		   ILSVRC2012_img_test.tar
		   ILSVRC2012_devkit.tar

	()	Within this folder, create the following hierarchy:
			data/images/train/ : content of ILSVRC2012_img_train.tar
			data/images/val/ : content of ILSVRC2012_img_val.tar
			data/images/test/ : content of ILSVRC2012_img_test.tar
			data/ILSVRC2012_devkit : content of ILSVRC2012_devkit.tar
	
	()	Save the downloaded images into data/imagenet12
	
	()	Preprocess all images to have height 256 pixels using the
		utils/preprocess-imagenet.sh script.
	
	()	Copy the data into RAM (if it fits)
	
	() Point the training code to the preprocessed data.
	
	() (Optional) Compile MatConvNet with GPU support. See
		http://www.vlfeat.org/matconvnet/ for instructions.
	
	() Run this file.
%}
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
%	Setup Options
% -------------------------------------------------------------------------

run(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab', 'vl_setupnn.m'));

opts.dataDir = fullfile('data','ILSVRC2012');
opts.modelType = 'dropout'; % bnorm or dropout
opts.expDir = fullfile('data', sprintf('imagenet12-%s', opts.modelType));

switch opts.modelType
	case 'dropout', opts.train.learningRate = logspace(-2, -4, 75);
	case 'bnorm',   opts.train.learningRate = logspace(-1, -3, 20);
end
opts.train.numEpochs = numel(opts.train.learningRate);

opts.numFetchThreads = 12;
opts.lite = false;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256;
opts.train.numSubBatches = 1;
opts.train.continue = true;
opts.train.gpus = [];
opts.train.prefetch = true;
opts.train.sync = true;
opts.train.expDir = opts.expDir;

% -------------------------------------------------------------------------
%	Initialise Data Storage Directory
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
	imdb = load(opts.imdbPath);
else
	imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite);
	mkdir(opts.expDir);
	save(opts.imdbPath, '-struct', 'imdb');
end

% -------------------------------------------------------------------------
%	Network Initialization
% -------------------------------------------------------------------------

% initialise model % REPLACE WITH IMPLEMENTATION FOR CUSTOMISATION
switch opts.modelType
	case 'dropout'
		net = cnn_imagenet_init();
 	case 'bnorm'
		net = cnn_imagenet_init_bnorm();
  	otherwise
		error('Unknown model type %s', opts.modelType);
end

bopts = net.normalization;
bopts.numThreads = opts.numFetchThreads;

% compute image statistics (mean, RGB covariances, etc)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat');
if exist(imageStatsPath)
  	load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance');
else
  	[averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts)
  	save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance');
end

% One can use the average RGB value, or use a different average for each pixel
%net.normalization.averageImage = averageImage;
net.normalization.averageImage = rgbMean;

% -------------------------------------------------------------------------
% 	Stochastic Gradient Descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance);
bopts.transformation = 'stretch';
bopts.averageImage = rgbMean;
bopts.rgbVariance = 0.1*sqrt(d)*v';
fn = getBatchWrapper(bopts);

% train
[net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true);




















% -------------------------------------------------------------------------
% 	Functions
% -------------------------------------------------------------------------

function fn = getBatchWrapper(opts)
	fn = @(imdb,batch) getBatch(imdb,batch,opts);

function [im,labels] = getBatch(imdb, batch, opts)
	images = strcat([imdb.imageDir filesep], imdb.images.name(batch));
	im = cnn_imagenet_get_batch(images, opts, 'prefetch', nargout == 0);
	labels = imdb.images.label(batch);

function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
	train = find(imdb.images.set == 1);
	train = train(1: 101: end);
	bs = 256;
	fn = getBatchWrapper(opts);
	for t=1:bs:numel(train)
		batch_time = tic;
		batch = train(t:min(t+bs-1, numel(train)));
		fprintf('collecting image stats: batch starting with image %d ...', batch(1));
		temp = fn(imdb, batch);
		z = reshape(permute(temp,[3 1 2 4]),3,[]);
		n = size(z,2);
		avg{t} = mean(temp, 4);
		rgbm1{t} = sum(z,2)/n;
		rgbm2{t} = z*z'/n;
		batch_time = toc(batch_time);
		fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time);
	end
	averageImage = mean(cat(4,avg{:}),4);
	rgbm1 = mean(cat(2,rgbm1{:}),2);
	rgbm2 = mean(cat(3,rgbm2{:}),3);
	rgbMean = rgbm1;
	rgbCovariance = rgbm2 - rgbm1*rgbm1';