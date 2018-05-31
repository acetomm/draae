import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

def triplet_finding(train_dir, k_similar, tfile_name):
    mnist = input_data.read_data_sets(train_dir=train_dir, validation_size=0)
    similar = []
    for i in range(mnist.train.images.shape[0]):
        a1 = mnist.train.images[i].reshape((1, 784))
        a1_label = mnist.train.labels[i]
        c_idx = np.argwhere(mnist.train.labels != a1_label)
        c1 = mnist.train.images[c_idx]
        c1 = c1.reshape((c1.shape[0], c1.shape[2]))
        mse = np.mean(np.square(a1 - c1), 1)
        mse_s = mse.argsort()
        similar.append(c_idx[mse_s][:k_similar].flatten())
        #print(mse[:5])
        #print(mse[mse_s][:5])
        #print(c_idx[mse_s][:5].flatten())
        #print(mse.shape, a1.shape, c1.shape)
        print('\r*triplet finding: {0} / {1}'.format(
                i, mnist.train.images.shape[0])
        , end='', flush=True)
    with open(tfile_name, 'wb') as f:
        f.write(pickle.dumps(similar))

class TwinMnist:
    def __init__(self, **tf_mnist_args):
        self.mnist = input_data.read_data_sets(**tf_mnist_args)
        self.nums_idx = []
        self.similar = None
        for i in range(10):
            n_idx = np.argwhere(self.mnist.train.labels == i)
            self.nums_idx.append(
                np.reshape(n_idx, (n_idx.size,))
            )

    def next_batch_couples(self, batch_size):
        nums = np.random.choice(10, size=batch_size)
        x1, x2 = [], []
        for i in nums:
            pair_idx = np.random.choice(self.nums_idx[i], size=2, replace=False)
            x1.append(self.mnist.train.images[pair_idx[0]])
            x2.append(self.mnist.train.images[pair_idx[1]])
        return np.array(x1), np.array(x2)

    def next_batch_triplets(self, batch_size):
        if self.similar is None:
            raise Exception('Triplets not loaded')
        nums = np.random.choice(10, size=batch_size)
        x1, x2, xc = [], [], []
        for i in nums:
            pair_idx = np.random.choice(self.nums_idx[i], size=2, replace=False)
            cx_idx = np.random.choice(self.similar[pair_idx[0]])
            x1.append(self.mnist.train.images[pair_idx[0]])
            x2.append(self.mnist.train.images[pair_idx[1]])
            xc.append(self.mnist.train.images[cx_idx])
        return np.array(x1), np.array(x2), np.array(xc)
    
    def sample(self, d, n):
        d_idx = np.random.choice(self.nums_idx[d], size=n, replace=False)
        d = self.mnist.train.images[d_idx]
        return d
    
    def load_triplets(self, tfile_name):
        with open(tfile_name, 'rb') as f:
            self.similar = pickle.loads(f.read())

def test_triplets(n):
    tm = TwinMnist(train_dir='MNIST_data', validation_size=0)
    x1, x2, xc = tm.next_batch_triplets(n)
    gs = gridspec.GridSpec(n, 3)
    for i in range(n):
        ax = plt.subplot(gs[i, 0])
        ax.imshow(x1[i].reshape((28, 28)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(gs[i, 1])
        ax.imshow(x2[i].reshape((28, 28)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(gs[i, 2])
        ax.imshow(xc[i].reshape((28, 28)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    plt.clf()


def test_sample(d, n):
    tm = TwinMnist(train_dir='MNIST_data', validation_size=0)
    ones = tm.sample(d, n)
    gs = gridspec.GridSpec(1, n, top=1., bottom=0., right=1., left=0., hspace=0.1, wspace=0.1)
    i = 0
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(ones[i].reshape((28, 28)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        i += 1
    plt.show()
    plt.clf()

def test_couples(n):
    tm = TwinMnist(train_dir='MNIST_data', validation_size=0)
    x1, x2 = tm.next_batch_couples(n)
    gs = gridspec.GridSpec(n, 2, top=1., bottom=0., right=1., left=0., hspace=0.1, wspace=0.1)
    #i = 0
    for i in range(n):
        ax = plt.subplot(gs[i, 0])
        ax.imshow(x1[i].reshape((28, 28)), cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(gs[i, 1])
        ax.imshow(x2[i].reshape((28, 28)), cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    plt.clf()

if __name__ == '__main__':
    #test_sample(0, 8)
    #test_couples(10)
    #test_triplets(10)
    #tm = TwinMnist(train_dir='MNIST_data', validation_size=0)
    
    tfile_name = 'mnist_triplets_k100'
    #triplet_finding('MNIST_data', 100, tfile_name)
    
    ''' '''
    tm = TwinMnist(train_dir='MNIST_data', validation_size=0)
    tm.load_triplets(tfile_name)
    
    n = 5
    x1, x2, xc = tm.next_batch_triplets(n)
    gs = gridspec.GridSpec(n, 3)
    for i in range(n):
        ax = plt.subplot(gs[i, 0])
        ax.imshow(x1[i].reshape((28, 28)), cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(gs[i, 1])
        ax.imshow(x2[i].reshape((28, 28)), cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(gs[i, 2])
        ax.imshow(xc[i].reshape((28, 28)), cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    plt.clf()
    
    
    #print(tm.similar)

