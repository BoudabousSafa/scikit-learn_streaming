from numpy import *

from scklearn.streams.cluster.DenStream import DenStream
from scklearn.streams.test.prequential_evaluation import prequential_evaluation
from sklearn import cluster, datasets

n_samples= 3000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

random.seed(0)

print('Load data')
import pandas as pd


S_Speed = 1000
N_train = 1000

#df = pd.read_csv("../../data/kddcup.csv")
#data = delete(df.values[:50000], [1,2,3,6,11,20,21], axis = 1)

Y = concatenate((noisy_circles[1], noisy_moons[1], blobs[1]), axis=0)
X = concatenate((noisy_circles[0], noisy_moons[0], blobs[0]), axis=0)
T = len(X)

print("Experimentation")

h = [DenStream()]

#E_pred,E_time = prequential_evaluation(X,Y,h,N_train)
E_time = prequential_evaluation(X,Y,h,N_train)

print("Evaluation")

E = zeros((len(h),T-N_train))
#for m in range(len(h)):
#    E[m] = get_errors(Y[N_train:], E_pred[m],J=Exact_match)

print("Plot Results")
from matplotlib.pyplot import *

fig, ax = subplots(2)
w = 200
for m in range(len(h)):
    acc = mean(E[m,:])
    time = mean(E_time[m,:])
    print(h[m].__class__.__name__)
    print("Exact Match %3.2f" % mean(acc))
    print("Running Time  %3.2f" % mean(time))
    print("---------------------------------------")
    acc_run = convolve(E[m,:], ones((w,))/w,'same')#[(w-1):]
    ax[0].plot(arange(len(acc_run)),acc_run, "-", label=h[m].__class__.__name__)
    acc_time = convolve(E_time[m,:], ones((w,))/w,'same')
    ax[1].plot(arange(len(acc_time)),acc_time, ":", label=h[m].__class__.__name__)

ax[0].set_title("Accuracy (exact match)")
ax[1].set_title("Running Time (ms)")
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
savefig("lab2_fig.pdf")
show()

