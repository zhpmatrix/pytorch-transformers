import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def load_data():
    digits = datasets.load_digits(n_class = 6)
    X,y = digits.data, digits.target
    return X,y

def load_data_(data_dir='event/'):
    X = np.load(data_dir+'/X.npy')
    y = np.load(data_dir+'/y.npy')
    return X, y

def transform(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=666)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def norm(X):
    x_min, x_max = X.min(0), X.max(0)
    X_norm = (X - x_min)/(x_max - x_min)
    return X_norm

def show(X,y):
    plt.figure(figsize=(8,8))
    for i in range(X.shape[0]):
        plt.text(X[i,0],X[i,1],str(y[i]),color=plt.cm.Set1(y[i]))
    plt.show()
    plt.savefig('fig.png')

if __name__ == '__main__':
    X,y = load_data()
    X,y = load_data_()
    X_tsne = transform(X)
    X_tsne_norm = norm(X_tsne)
    show(X_tsne_norm, y)
