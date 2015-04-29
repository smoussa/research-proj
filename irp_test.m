% run the CNN
res = vl_simplenn(net,im_);

% show the classification result
scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1); clf; imagesc(im);
title(sprintf('%s (%d), score %.3f', net.classes.description{best}, best, bestScore));