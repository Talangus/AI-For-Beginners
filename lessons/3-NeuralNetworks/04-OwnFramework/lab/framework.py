import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self,nin,nout):
        self.W = np.random.normal(0, 1.0/np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1,nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x):
        self.x=x
        return np.dot(x, self.W.T) + self.b
    
    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        self.dW = dW
        self.db = db
        return dx
    
    def update(self,lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db

class Softmax:
    def forward(self,z):
        self.z = z
        zmax = z.max(axis=1,keepdims=True)
        expz = np.exp(z-zmax)
        Z = expz.sum(axis=1,keepdims=True)
        return expz / Z
    def backward(self,dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)
    
class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()
    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p
    
class Net:
        def __init__(self):
            self.layers = []
        
        def add(self,l):
            self.layers.append(l)
            
        def forward(self,x):
            for l in self.layers:
                x = l.forward(x)
            return x
        
        def backward(self,z):
            for l in self.layers[::-1]:
                z = l.backward(z)
            return z
        
        def update(self,lr):
            for l in self.layers:
                if 'update' in l.__dir__():
                    l.update(lr)

class Tanh:
    def forward(self,x):
        y = np.tanh(x)
        self.y = y
        return y
    def backward(self,dy):
        return (1.0-self.y**2)*dy

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    def backward(self, dy):
        return dy * (self.x > 0)
    
class Sigmoid:
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    def backward(self, dy):
        return dy * self.y * (1 - self.y)

def get_loss_acc(x, y, net,loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p,y)
    pred = np.argmax(p,axis=1)
    acc = (pred==y).mean()
    return l,acc

def train_epoch(net, train_x, train_y, test_x, test_y, loss=CrossEntropyLoss(), batch_size=4, lr=0.1, monitor_acc=False):
    num_of_iterations = len(train_x) / batch_size
    history = {
        "iteration": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "weights": [],
    }
    for i in range(0,len(train_x),batch_size):
        xb = train_x[i:i+batch_size]
        yb = train_y[i:i+batch_size]

        p = net.forward(xb)
        l = loss.forward(p,yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr)

        if monitor_acc and (i % (100 * batch_size) == 0 or i + batch_size >= len(train_x)):
            current_iter = i // batch_size + 1
            train_l, train_a = get_loss_acc(train_x, train_y, net=net)
            val_l, val_a = get_loss_acc(test_x, test_y, net=net)
            history["iteration"].append(current_iter)
            history["train_loss"].append(train_l)
            history["train_acc"].append(train_a)
            history["val_loss"].append(val_l)
            history["val_acc"].append(val_a)
            max_weights = []
            for l in net.layers:
                if hasattr(l, "W"):
                    max_weights.append(np.abs(l.W).max())
            history["weights"].append(max_weights)
    
    return history

def plot_train_val_history(history):
    import matplotlib.pyplot as plt
    import numpy as np

    train_acc_percent = np.array(history["train_acc"]) * 100
    val_acc_percent = np.array(history["val_acc"]) * 100
    acc_diff_percent = np.abs(train_acc_percent - val_acc_percent)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history["iteration"], train_acc_percent, label="Train Accuracy")
    plt.plot(history["iteration"], val_acc_percent, label="Validation Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy over Iterations")
    plt.ylim(0, 100)

    plt.subplot(1,2,2)
    plt.plot(history["iteration"], acc_diff_percent, label="|Train - Validation|")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy Difference (%)")
    plt.legend()
    plt.title("Accuracy Difference (Train - Validation)")
    plt.ylim(0, 5)

    plt.tight_layout()
    plt.show()
    print(f"Final Validation Accuracy: {val_acc_percent[-1]:.2f}%")

def plot_weights_history(history):
    import matplotlib.pyplot as plt
    weights = np.array(history["weights"])  
    iterations = history["iteration"]
    plt.figure(figsize=(8,5))
    for i in range(weights.shape[1]):
        plt.plot(iterations, weights[:,i], label=f"Layer {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Max |Weight|")
    plt.title("Max Absolute Value of Weights per Layer During Training")
    plt.legend()
    plt.show()