from sklearn.base import BaseEstimator
import numpy as np
import scipy.special import expit as sigmoid


class LogReg(BaseEstimator):
    def __predict__(self, X):
        return 1/(1+np.exp(X.dot(w)))

    def __loss__(self, X, y):
        return np.mean(np.log(1 + np.exp(-y * (X.dot(self.w))))) + \
                self.lambda_2*np.sum(self.w * self.w)

    def __init__(self, lambda_1=0.01, lambda_2=1.0, gd_type='full',
                 tolerance=1e-4, max_iter=10, w0=None, alpha=1e-3,
                 verbose=False, batch_size=12):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.loss_history = None
        self.verbose = verbose
        self.batch_size = batch_size

        if gd_type == 'full':
            self.__grad__ = self.calc_gradient
            self.__loss__ = self.calc_loss
        elif gd_type == 'stochastic':
            self.__grad__ = self.calc_gradient_stohastic
            self.__loss__ = self.calc_loss
        elif gd_type == 'stochastic_loss':
            self.__grad__ = self.calc_gradient_stohastic
            self.__loss__ = self.calc_loss_stohastic

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        if self.w0 is not None:
            self.w = self.w0
        else:
            self.w = np.zeros(X.shape[1])

        last_loss = 0
        for epoch in range(self.max_iter):
            self.w -= self.alpha * self.__grad__(X, y)
            current_loss = self.__loss__(X, y)
            self.loss_history.append(current_loss)
            if np.abs(current_loss-last_loss) <= self.tolerance:
                if self.verbose:
                    sys.stderr.write("Finish on %d iterasion" % epoch)
                return self
            last_loss = current_loss

        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return np.array([[1.0-p, p] for p in sigmoid(X.dot(self.w))])

    def calc_gradient_stohastic(self, X, y):
        batch = sorted(np.random.choice(X.shape[0], size=self.batch_size))
        # PEP8 :( i don't know how to break math lines
        a, b = X[batch], y[batch]
        return a.T.dot(-1.0 / (1.0 + np.exp(b * a.dot(self.w))) * b) / \
            b.shape[0] + \
            self.lambda_2*self.w

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        """ Not np realization
        dw = np.zeros(X.shape[1])
        for yi, x in zip(y,X):
            dw+=-yi*x/(1.0+math.exp(yi*np.dot(self.w,x)))
        return dw+self.lambda_2*self.w
        """

        return X.T.dot(-1.0 / (1.0 + np.exp(y * X.dot(self.w))) * y) / \
            y.shape[0] + self.lambda_2*self.w

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        """ Not np realization
        ress = 0
        for yi, x in zip(y,X):
            ress+=math.log(1+math.exp(-yi*np.dot(self.w,x)))
        return ress+alpha2/2.0*np.dot(self.w,self.w)
        """
        return np.mean(np.log(1+np.exp(-y*(X.dot(self.w))))) + \
            self.lambda_2/2.0*np.dot(self.w, self.w)

    def calc_loss_stohastic(self, X, y):
        batch = sorted(np.random.choice(X.shape[0], size=self.batch_size))
        X_batch, y_batch = X[batch], y[batch]
        return np.mean(np.log(1+np.exp(-y_batch*(X_batch.dot(self.w))))) + \
            self.lambda_2/2.0*np.dot(self.w, self.w)
