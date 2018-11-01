function [B1,B2] = hash_code_gen(train_x, test_x, codelength, method_i)

TrainNum=size(train_x,1);
Ndim=size(train_x,2);
TestNum=size(test_x,1);
randNum=randperm(TrainNum);

switch(method_i)
    case 'LSH'
        tic
        W = randn(Ndim,codelength);
        toc
        B1 = (train_x*W)>0;
        B2 = (test_x*W)>0;
        B1=compactbit(B1);
        B2=compactbit(B2);
        
    case 'SH'
        addpath(genpath('.\hash_toolbox\spectral_hashing'));
        SHparam.nbits = codelength;
        tic
        SHparam = trainSH(train_x, SHparam);
        toc
        [B1,U1] = compressSH(train_x, SHparam);
        [B2,U2] = compressSH(test_x, SHparam);
       
        
    case 'USPLH'
        addpath(genpath('.\hash_toolbox\SPLH_release'));
        USPLHparam.nbits=codelength;
        USPLHparam.eta=0.125;
        tic
        USPLHparam = trainUSPLH(train_x, USPLHparam);
        toc
        [B1, U1] = compressUSPLH(train_x, USPLHparam);
        [B2, U2] = compressUSPLH(test_x, USPLHparam); 
        
        
    case 'AGH'
        addpath('.\hash_toolbox\Anchor_Graph_Hash');
        tic
        AnchorNum=500;
        Anchors=GetAnchors(train_x,AnchorNum,0.01);
        [Y, W, sigma] = OneLayerAGH_Train(train_x, Anchors, codelength, 100, 0);
        toc
        tY = OneLayerAGH_Test(test_x, Anchors, W, 100, sigma);
        B1=compactbit(Y>0);
        B2=compactbit(tY>0);
        
    case 'PCAH'
        tic
        [pc, l] = eigs(cov(train_x),codelength);
        toc     
        B1=compactbit((train_x*pc)>0);
        B2=compactbit((test_x*pc)>0);
        
    case 'ITQ'
        addpath('.\hash_toolbox\ITQ_release');
        Fea=double([train_x; test_x]);
        clear train_x test_x
        sampleMean = mean(Fea,1);
        Fea = (Fea - repmat(sampleMean,size(Fea,1),1));
        tic
        % PCA
        [pc, l] = eigs(cov(Fea(1:TrainNum,:)),codelength);
        Fea_pca = Fea * pc;
        % ITQ
        [B, R] = ITQ(Fea_pca,50);
        Fea_pca = Fea_pca*R;
        toc
        B = zeros(size(Fea_pca));
        B(Fea_pca>=0) = 1;
        B = compactbit(B>0);
        B1=B(1:TrainNum, :);
        B2=B((TrainNum+1):end,:);
        
    case 'IsoHash'
        addpath('.\hash_toolbox\IsoHash');
        tic
        [pc,Lambda]=eigsdescend(cov(double(train_x)),codelength);
        Q=GradientFlow(Lambda);
        pc=pc*Q;
        toc
        B1=compactbit((train_x*pc)>0);
        B2=compactbit((test_x*pc)>0);
        
    case 'SpH'
        addpath('.\hash_toolbox\Spherical_Hashing_Src_Matlab');
        tic
        [centers, radii] = SphericalHashing( train_x, codelength );
        dData = distMat( [train_x;test_x] , centers );
        toc
        th = repmat( radii' , size( dData , 1 ) , 1 );
        bData = zeros( size(dData) );
        bData( dData <= th ) = 1;
        bData = compactbit(bData);
        B1=bData(1:TrainNum,:);
        B2=bData((TrainNum+1):end,:);

    case 'IMH'
        addpath(genpath('.\hash_toolbox\IMH_release-master'));
        tic
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
        EmbeddingX = Z*mappedX;
        toc
        B1 = EmbeddingX > 0;
        [tZ] = get_Z(test_x, Anchors,  options.s, sigma);
        tEmbedding = tZ*mappedX;
        B2 = tEmbedding > 0;
        B1=compactbit(B1);
        B2=compactbit(B2); 
          
end

