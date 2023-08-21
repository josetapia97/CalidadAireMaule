import mxnet as mx
from mxnet import gluon,np,autograd
from read_data_mxnet import *

class Model(gluon.HybridBlock):

    def __init__(self,output_size):
        super(Model, self).__init__()
        self.rnn1 = gluon.rnn.LSTM(20)
        self.rnn2 = gluon.rnn.LSTM(20)
        self.mydense = gluon.nn.Dense(output_size)

    def forward(self, x):
        x = mx.npx.relu(self.rnn1(x))
        x = mx.npx.relu(self.rnn2(x))
        return mx.npx.relu(self.mydense(x))

context_length=6
prediction_length=6
(X_train,y_train),(X_test,y_test)=create_dataset_ucm_train(6,prediction_length),create_dataset_lf_test(6,prediction_length)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

device = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
net=Model(prediction_length)
net.initialize(ctx=device)
net.hybridize()

data_mean = np.mean(X_train).astype('float32')
data_std = np.std(X_train).astype('float32')
label_mean = np.mean(y_train).astype('float32')
label_std = np.std(y_train).astype('float32')

def transform(data, label):
    #data = (data.astype('float32')-mean)/std
    data = (np.array(data).astype('float32')-data_mean)/data_std
    label = (np.array(label).astype('float32')-label_mean)/label_std
    return data, label

train_dataset = mx.gluon.data.dataset.ArrayDataset(X_train, y_train).transform(transform)
test_dataset = mx.gluon.data.dataset.ArrayDataset(X_test, y_test).transform(transform)

batch_size = 64
train_loader = gluon.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = gluon.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
train_mse = gluon.metric.MSE()
test_mse = gluon.metric.MSE()
loss = gluon.loss.L2Loss()

for epoch in range(50):
    train_loss=0.0
    for data, label in train_loader:
        data=data.to_device(device)
        label=label.to_device(device)
        # forward + backward
        with autograd.record():
            output = net(data)
            loss_val = loss(output, label)
        #autograd.backward(loss)
        loss_val.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss_val.mean().asnumpy()
        train_mse.update(labels = label, preds = output)
    # calculate validation accuracy
    for data, label in test_loader:
        data=data.to_device(device)
        label=label.to_device(device)
        test_mse.update(labels = label, preds = net(data))
    print("Epoch %d: train loss %.6f, train rmse %.6f, test rmse %.6f" % (
        epoch, train_loss/len(train_loader),train_mse.get()[1],test_mse.get()[1]))

file_name = "net.params"
net.save_parameters(file_name)