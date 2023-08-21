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
(X_test,y_test)=create_dataset_test(context_length,prediction_length)

device = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
net=Model(prediction_length)
net.load_parameters('net.params', ctx=device)
net.hybridize()

data_mean = np.mean(X_test).astype('float32')
data_std = np.std(X_test).astype('float32')
label_mean = np.mean(y_test).astype('float32')
label_std = np.std(y_test).astype('float32')

def transform(data, label):
    #data = (data.astype('float32')-mean)/std
    data = (np.array(data).astype('float32')-data_mean)/data_std
    label = (np.array(label).astype('float32')-label_mean)/label_std
    return data, label

test_dataset = mx.gluon.data.dataset.ArrayDataset(X_test, y_test).transform(transform)

batch_size = 64
test_loader = gluon.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_mse = gluon.metric.MSE()
loss = gluon.loss.L2Loss()


for data, label in test_loader:
    data=data.to_device(device)
    label=label.to_device(device)
    test_mse.update(labels = label, preds = net(data))
print(" test rmse %.6f "%(test_mse.get()[1]))