function [B1,B2] = HashRecon(train_x,test_x,codelength, HashMethod)

TrainNum=size(train_x,1);
Ndim=size(train_x,2);

switch(HashMethod)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = LSH
        case 'LSH'
        W = randn(Ndim,codelength);
        TTr = train_x*W;
        clear W

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = SH
        case 'SH'
        addpath(genpath('.\hash_toolbox\spectral_hashing\'));
        SHparam.nbits = codelength;
        SHparam = trainSH(train_x, SHparam);
        [B1,TTr] = compressSH(train_x, SHparam);
        clear B1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = USPLH
        case 'USPLH'
        addpath(genpath('.\hash_toolbox\SPLH_release\USPLH\'));
        USPLHparam.nbits=codelength;
        USPLHparam.eta=0.125;
        USPLHparam = trainUSPLH(train_x, USPLHparam);
        [B1, TTr] = compressUSPLH(train_x, USPLHparam);
        clear B1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = AGH
        case 'AGH'
        addpath('.\hash_toolbox\Anchor_Graph_Hash');
        AnchorNum=500;
        Anchors=GetAnchors(train_x,AnchorNum,0.01);
        [TTr, W, sigma] = OneLayerAGH_Train(train_x, Anchors, codelength, 100, 0);
        clear Anchors W sigma

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = PCAH
        case 'PCAH'
        [pc, l] = eigs(cov(train_x),codelength);
        TTr = train_x*pc;
        clear pc l

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = ITQ
        case 'ITQ'
        addpath('.\hash_toolbox\ITQ_release');
        % PCA
        [pc, l] = eigs(cov(train_x),codelength);
        Fea_pca = train_x * pc;
        % ITQ
        [B, R] = ITQ(Fea_pca,50);
        TTr = Fea_pca*R;
        clear pc l Fea_pca B R

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = IsoHash
        case 'IsoHash'
        addpath('.\hash_toolbox\IsoHash');
        [pc,Lambda]=eigsdescend(cov(train_x),codelength);
        Q=GradientFlow(Lambda);
        pc=pc*Q;
        TTr = train_x*pc;
        clear pc Lambda Q

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = SpH
        case 'SpH' 
        addpath('.\hash_toolbox\Spherical_Hashing_Src_Matlab');
        [centers, radii] = SphericalHashing( train_x, codelength );
        dData = distMat( train_x, centers );
        th = repmat( radii' , size( dData , 1 ) , 1 );
        TTr = dData - th;
        clear centers radii dData th

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hash method = IMH
        case 'IMH'
        addpath(genpath('.\hash_toolbox\IMH_release-master'));
        AnchorNum=500;
        Anchors=GetAnchors(train_x,AnchorNum,0.01);
        % Set parameters
        options.s = 5;
        options.sigma = 0;
        no_dims = codelength;
        initial_dims = 50;
        perplexity = 30;
        mappedX = tsne(Anchors, [], no_dims, initial_dims, perplexity);
        [Z,~, sigma] = get_Z(train_x, Anchors, options.s, options.sigma);
        TTr = Z*mappedX;
        clear Anchors options mappedX Z
end


%%%%%%%%%%%%%% Hash code reconstruction
% TTr=TTr';
% %%% Normalize data to have zero mean per pixel
% data.mean = mean(TTr,2);
% TTr = TTr - data.mean * ones(1,size(TTr,2));
% %%% Make mean variance to be 1.
% data.scale  = 1/mean(std(TTr,0,2));
% TTr = TTr * data.scale;
% TTr=TTr';

sampleMean = mean(TTr,1);
TTr = (TTr - repmat(sampleMean,size(TTr,1),1));

InputNod = codelength;
HiddenNod = codelength;
tic
InputWeight = rand(InputNod,HiddenNod)*2-1;
bias = rand(HiddenNod,1)'*2-1 ;
BiasMatrix = repmat( bias, TrainNum, 1 );
H_train = TTr*InputWeight + BiasMatrix;
H_train = 1 ./ (1 + exp(-H_train));
OutputWeight = (pinv(H_train) * train_x)'; 
toc

%%%%%%%%% Generate the final codes
train_h = train_x * OutputWeight;
test_h = test_x * OutputWeight;
data = zscore([train_h; test_h]);
train_h = data(1:TrainNum,:);
test_h = data (TrainNum+1:end,:);
B1 = (train_h>0.5);
B2 = (test_h>0.5);
B1=compactbit(B1);
B2=compactbit(B2);


















