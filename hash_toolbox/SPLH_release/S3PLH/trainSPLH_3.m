function SPLHparam = trainSPLH_3(X, SPLHparam,vad_Data,SR_M)

% Input
%  Sequential Projection Learning Based Hashing
%  USPLHparam.nbits = number of bits (nbits do not need to be a multiple of 8)
%
%  By Jun Wang (jwang@ee.columbia.edu)
%  Initial version July, 2009
%  Following the code style from the following paper
% "spectral hashing", nips 2008
% Last update Jan. 20, 2010

%%% number and dim of data
[Nsamples Ndim] = size(X);

% %% Remove sample mean
sampleMean = mean(X,1);

%%% Store original data
ori_X=X;
nbits = SPLHparam.nbits;

%%% size of the semantic relevance matrix
srm_num=size(SR_M,1);
pca_options.ReducedDim=10;
maskS_pre=zeros(srm_num,srm_num);
for i_bit=1:SPLHparam.nbits
    %%% Assume Nsamples>>Ndim  for a realistic large scale problem
    covdata = X'*X;
    covdata=covdata/(Nsamples);     
    C_M=vad_Data*SR_M*vad_Data'/srm_num;
    M=covdata+SPLHparam.eta*C_M;
    [eigvector, eigvalue, elapse] = PCA2(M, pca_options);
    for ii=1:10    
       deltaS=vad_Data'*eigvector(:,ii)*eigvector(:,ii)'*vad_Data;
       maskS=sign(SR_M.*deltaS);
       aa(ii)=sum(sum((maskS_pre>0).*(maskS<0)));
       bb(ii)=sum(sum((maskS_pre<0).*(maskS>0)));
    end
    [E1,IX1]=sort(aa,'ascend');
    [E2,IX2]=sort(bb,'descend');
    cc=aa-bb;
    [E,IX]=sort(cc,'ascend');
    w(:,i_bit)=eigvector(:,IX(1));
    deltaS=vad_Data'*w(:,i_bit)*w(:,i_bit)'*vad_Data;
    maskS_pre=sign(SR_M.*deltaS);
    T_deltaS=(maskS_pre<0).*deltaS;
    SR_M=SR_M-T_deltaS;
    projectX(:,i_bit) = X * w(:,i_bit);
    X=X-projectX(:,i_bit)*w(:,i_bit)';      
    %vad_Data=(vad_Data'-vad_Data'*eigvector*eigvector')';   
end

b=zeros(1,nbits);

% 4) store paramaterswwwSPLHparam.mean=sampleMean;
SPLHparam.w = w;
SPLHparam.b = zeros(1,nbits);

%SPLHparam.projected=projectX;
%save(['SPLHparam_' num2str(nbits) '.mat'],'SPLHparam');
