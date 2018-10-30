function score=precision(Dtrue, Dham, Deu, NN)

[TestNum, TrainNum]=size(Dtrue);
score=0;
for i=1:TestNum
    
    DistNum=unique(Dham(i,:));
    len=0;
    j=1;
    while(j<=length(DistNum)&&len<1000)
        Idx=find(Dham(i,:)<=(DistNum(j)+0.00001));
        len=length(Idx);
        j=j+1;
    end
    flag=double(Dham(i,:)<=(DistNum(j-1)+0.00001));
    D=Deu(i,:).*flag;
    
    C=sort(D,'descend');
    D=double(D>=C(NN));
    
    score=score+sum(D.*Dtrue(i,:));
    
end
score=score/(TestNum*NN);