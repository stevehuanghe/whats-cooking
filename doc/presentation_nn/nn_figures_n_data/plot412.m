data = load('logs_ndp_bn.txt');

train_loss = data(:,1);
train_acc = data(:,2);
test_loss = data(:,3);
test_acc = data(:,4);
epoch = ( 1: 400);


figure(2)

subplot(2,1,1)
plot(epoch,train_acc,'r')
title('Neural Net without Dropout with Batch-norm: Accuracy')
xlabel('epoch')
ylabel('accuracy')
hold on
plot(epoch,test_acc,'b')
subplot(2,1,2)
plot(epoch,train_loss,'r')
title('Loss')
xlabel('epoch')
ylabel('loss')
hold on
plot(epoch,test_loss,'b')