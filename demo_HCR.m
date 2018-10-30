clc;
clear;
close all

% addpath('E:\research\code\DimensionReduction\t_SNE\tSNE_matlab');

%%%%%%%%%%%% Experimental settings
dataset = {'Tiny100K','GIST1M','CIFAR10','MNIST'};
dataset_i = char( dataset(1) );  %%% choose the dataset
method = {'LSH','SH','PCAH','USPLH','AGH','ITQ','SpH','IsoHash','IMH'};
loopbits = [16 32 48 64 96 128]; %%%%  set hash code length

%%%%%%%%% Load data
path = ['.\data_set\' dataset_i '.mat'];
load(path);
TrainNum=size(train_x,1);
TestNum=size(test_x,1);

%%%%%%%%%%% Normalize data to have zero mean
Fea=double([train_x; test_x]);
sampleMean = mean(Fea,1);
Fea = (Fea - repmat(sampleMean,size(Fea,1),1));
train_x=Fea(1:TrainNum, :);
test_x=Fea((TrainNum+1):end,:);
clear Fea

%%%%%%%%%%%%%%%%% Hash Code Learning 
for codelength=loopbits 
  for method_i=method
      fprintf('hash code length = %d\n',codelength);
      
      %%%%%%%%%%% Hash Code Learning with Original Hashing
      disp(['hash method = ' char(method_i)]);
      [B1,B2] = hash_code_gen(train_x,test_x,codelength,char(method_i));
      path = ['.\hash_code\' dataset_i '_' char(method_i) '_' num2str(codelength) 'bits.mat'];
      save(path,'B1','B2'); %%% save the hash codes
    
      %%%%%%%%%%% Hash Code Reconstruction with our HCR
      disp(['hash method = ' char(method_i) ' + HCR']);
      [B1,B2] = HashRecon(train_x,test_x,codelength, char(method_i));
      path = ['.\hash_code\' dataset_i '_' char(method_i) '_HCR_' num2str(codelength) 'bits.mat'];
      save(path,'B1','B2'); %%% save the hash codes
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Performance Evaluation
evl_method = {'LSH','SH','PCAH','USPLH','AGH','ITQ','SpH','IsoHash','IMH','LSH_HCR','SH_HCR','PCAH_HCR','USPLH_HCR','AGH_HCR','ITQ_HCR','SpH_HCR','IsoHash_HCR','IMH_HCR'};
for method_i = method
    
    disp(['evaluation method=' char(method_i)]);
    
    %%%%%%%% initialization
    MAP=zeros(length(loopbits),TestNum);
    ind = [1,5,10,20,30,50,80,100,100:10:TrainNum, TrainNum]; 
%     ind = [1,5,10,20,30,50,80,100,200,300,500,800,1000:1000:TrainNum];  %%%%%%% for GIST-1M dataset
    presicion_rank=zeros(length(loopbits),length(ind));
    recall_rank=zeros(length(loopbits),length(ind));
    presicion_lookup=zeros(length(loopbits),1);
    recall_lookup=zeros(length(loopbits),1);
    
    ii=0;
    for codelength=loopbits
        ii=ii+1;
        fprintf('codelength=%d\n',codelength);
        %%%%%%%% load the hash codes
        path = ['.\hash_code\' dataset_i '_' char(method_i) '_' num2str(codelength) 'bits.mat'];
        load(path);
        
        for j=1:TestNum
            Dhamm = hammingDist(B2(j,:), B1);
            [E,IX1]=sort(Dhamm, 2, 'ascend');
            IX2=groundtruth(j,:); %%%%%%%%%% for Euclidean groundtruth 
%             IX2=find(NeighborTrue(j,:)==1);  %%%%%%%%%% for label groundtruth
            sorted_truefalse=ismember(IX1, IX2);
            truepositive=cumsum(sorted_truefalse);
            
             %%%%%%%%%%%% Computing MAP score
             pos = find(sorted_truefalse>0);
             p=truepositive(pos)./(pos);
             MAP(ii,j) = mean(p);
             
             %%%%%%%%%%%% evaluation by Hamming ranking pre-rec
             presicion_rank(ii,:) = presicion_rank(ii,:) + truepositive(ind)./ind;
             recall_rank(ii,:) = recall_rank(ii,:) + truepositive(ind)/sum(sorted_truefalse);
             
             %%%%%%%%%%%% evaluation by Hash lookup(r<2)
             IX3 = Dhamm<(2+1e-5);
             %%%% for Euclidean groundtruth
             IX4 = zeros(1,length(IX3)); 
             IX4(groundtruth(j,:))=1;
             hit_num = double(IX3)*IX4'; 
             presicion_lookup(ii) = presicion_lookup(ii) + hit_num/(sum(IX3)+1e-10);
             recall_lookup(ii) = recall_lookup(ii) + hit_num/(sum(IX4)+1e-10);
%              %%%% for label groundtruth
%              hit_num = double(IX3)*NeighborTrue(j,:)'; 
%              presicion_lookup(ii) = presicion_lookup(ii) + hit_num/(sum(IX3)+1e-10);
%              recall_lookup(ii) = recall_lookup(ii) + hit_num/(sum(NeighborTrue(j,:))+1e-10);
        end
    end
    MAP = mean(MAP,2);
    presicion_rank = presicion_rank/TestNum;
    recall_rank = recall_rank/TestNum;
    presicion_lookup = presicion_lookup/TestNum;
    recall_lookup  = recall_lookup/TestNum;
    
    %%%%%% Save the evaluation results
    path=['.\evaluation_results\' dataset_i '_' char(method_i) '.mat'];
    save(path, 'MAP', 'presicion_rank', 'recall_rank', 'presicion_lookup','recall_lookup','loopbits', 'ind');
end

