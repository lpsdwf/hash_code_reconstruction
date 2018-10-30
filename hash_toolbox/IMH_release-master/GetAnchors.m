function [Anchors] = GetAnchors(X, AnchorNum, sample_rate)

[n,dim] = size(X);

SampleNum=round(n*sample_rate);

ind=randperm(n);
X_sub=X(ind(1:SampleNum),:);
clear X

Anchors=kmeans(X_sub,AnchorNum);