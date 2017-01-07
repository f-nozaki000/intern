import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import glob

from chainer import Chain
from chainer import iterators, optimizers, training
import chainer.links as L
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer.training import extensions

# number of data
n_data = 5

# number of epoch
n_epoch = 10

# blanks for correct parameters
bfcp = np.zeros([n_data, 3])


# make sin graphs & make an array of correct parameters
for i in range(0, n_data):
    x = np.linspace(0, 2*np.pi)
    r1 = random.random()
    r2 = random.uniform(1, 2)
    r3 = random.uniform(0, np.pi)

    graph = plt.plot(x, r1*np.sin(r2*(x+r3)))

    s = str(i)
    filename = s.rjust(4, '0')
    plt.savefig(filename)
    plt.show()

    cp = np.array([r1, (2*np.pi)/r2, r1*np.sin(r2*r3)])

    bfcp[i, :] = cp[:]
    params = bfcp.astype(np.float32)

print(bfcp)
print(bfcp.shape)
print(len(bfcp))

# make a blank list for arrays of graphs
list = []

# read files & delete column of transparency
for png in glob.glob('/home/fuka/PycharmProjects/graphtocode/' + '*.png'):
    im = Image.open(png)

    imtoimg = np.delete(np.array(im.resize((600, 600))), 3, 2)
    img = imtoimg.astype(np.float32).reshape(3, 600, 600)/255

    list.append(img)

print(len(list))

class CNNRegression(Chain):
    def __init__(self):
        super().__init__(
            conv1=L.Convolution2D(3, 32, 30, stride=10),
            conv2=L.Convolution2D(32, 32, 6, stride=2),
            l3=L.Linear(512, 256),
            l4=L.Linear(256, 3)
        )

    def __call__(self, x_data):

        # print(x_data.data.shape)
        h = F.max_pooling_2d(F.relu(self.conv1(x_data)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)
        return y

train = tuple_dataset.TupleDataset(list, params)
train_iter = iterators.SerialIterator(train, 5)

model = L.Classifier(CNNRegression(), lossfun=F.mean_squared_error)
model.compute_accuracy = False

optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())

trainer.run()
