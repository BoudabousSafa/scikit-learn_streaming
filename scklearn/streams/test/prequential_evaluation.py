from numpy import *
from time import clock


def exact(yt, yp):
    '''
        Error function
        --------------
    '''
    return (yp == yt) * 1


from ..test.metrics import J_index


def get_errors(Y, P, J=J_index):
    N, L = Y.shape
    E = zeros((N))
    for i in range(N):
        E[i] = J(Y[i, :].reshape(1, -1), P[i, :].reshape(1, -1))
    return E


def prequential_evaluation(X, Y, H, N_train):
    '''
        Prequential Evaluation
        ----------------------
        X                       instances
        Y                       labels
        H = [h_1,...,h_H]       a set of classifiers
        N_train                 number of instances for initial batch
        return the label predictions for each test instance, and the associated running time
    '''
    M = len(H)
    T = len(Y)

    # split off an initial batch (maybe) ...
    Y_init = Y[0:N_train]
    X_init = X[0:N_train]

    # ... and then use the remainder, used for both incremental training and evaluation.
    Y = Y[N_train:]
    X = X[N_train:]

    E_pred = zeros((M, T - N_train, 1))
    E_time = zeros((M, T - N_train))

    for m in range(M):
        # start_time = clock()
        H[m].fit(X_init)
        # E_time[m,0] = clock() - start_time

    for t in range(0, T - N_train):
        for m in range(M):
            start_time = clock()
            # E_pred[m,t,:] = H[m].predict(X[t,:].reshape(1,-1))
            print(t)
            H[m].partial_fit(X[t, :].reshape(1, -1), None)
            E_time[m, t] += (clock() - start_time)

    # return E_pred, E_time
    return E_time


def seq_prequential(Y, h, init_ratio=2, error_fn=exact):
    '''
        Sequence-Prequential Evaluation
        -------------------------------
        As opposed to standard data streams, the X space is optional!
        (we are dealing here with more of an auto-encoder / Markov process)
        predict_update must be called on each instance!

        X: stream input
        Y: stream output
        h: classifiers(s) to evaluate (may be a list of classifiers)
        returns the results
    '''

    T, L = Y.shape

    print("Initial fit")
    T_init = int(T / init_ratio)

    for h_ in h:
        h_.fit(Y[0:T_init])

    M = len(h)

    P = zeros((M, T, L))
    E = zeros((M, T, L))

    s = zeros((L))

    print("Stream preq.")
    for t in range(T_init, T):
        n = t - T_init
        # if n % (T-T_init)/100:
        #     print(".")
        for m in range(M):
            P[m, t] = h[m].predict_update(Y[t].reshape(1, -1))
            # E[m,t] = error_fn(Y[t],P[m,t])
            E[m, t] = error_fn(Y, P[m], t)
        s += (Y[t] > 0)

    P = P[:, T_init:, :]
    E = E[:, T_init:, :]

    return P, E, P.shape[1]
