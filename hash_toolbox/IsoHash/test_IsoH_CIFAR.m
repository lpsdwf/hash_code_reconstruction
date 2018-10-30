load CIFAR10

train_x = double(train_x);
test_x = double(test_x);

[n, d] = size(train_x);
tn = size(test_x,1);
mvec = mean(train_x);
train_x = train_x - repmat(mvec, n, 1);
test_x = test_x - repmat(mvec, tn, 1);

loopbits = [8 16 24 32 64];

for nbits = loopbits    
    tic
    [pc,Lambda]=eigsdescend(cov(train_x),nbits);
    Q=GradientFlow(Lambda);
    pc=pc*Q;
    Y = train_x * pc > 0;
    tY = test_x * pc > 0;
    save(['CIFAR_result2\Y_', num2str(nbits), 'bits'], 'Y');
    save(['CIFAR_result2\tY_', num2str(nbits), 'bits'], 'tY');
    toc
end
Evaluate_CIFAR;