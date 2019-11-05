import sys
import numpy as np
import pandas as pd
from Blackbox31 import blackbox31
from Blackbox32 import blackbox32

# -------------------------------------------------------
blackbox = sys.argv[-1]
if blackbox == "blackbox31":
    bb = blackbox31
    name = "blackbox31"
elif blackbox == "blackbox32":
    bb = blackbox32
    name = "blackbox32"
# -------------------------------------------------------


class StoreData:

    def __init__(self):

        global bb
        global name
        self.bb = bb
        self.name = name
        self.test_size = 200
        self.train_size = 1000

        self.labels = []  # Unique labels
        self.label_count = {}  # Count of labels seen
        self.p_label = {}  # Prob of labels seen

        self.avg_matrix = {}  # Dict, Key:(label, feature index), Value: Average
        self.var_matrix = {}  # Dict, Key:(label, feature index), Value: Variance
        self.old_var = {}  # Dict, Key:(label, feature index), Value: Previous value seen

    # -------------------------------------------------------

    def input(self):

        """
        1. Read Test Data
        2. Read Train Data
        3. Drive the classifier class to predict
        :return:
        """

        X_test, y_test = [0 for _ in range(self.test_size)], [0 for _ in range(self.test_size)]

        for i in range(self.test_size):
            x, y = self.bb.ask()
            X_test[i] = x
            y_test[i] = y

        X_test = pd.DataFrame(X_test)

        # ---------------------------------------
        # Init accuracy matrix to plot increase in accuracy
        accuracy = []
        # ---------------------------------------

        X_train, y_train = [], []

        for i in range(self.train_size):
            x, y = self.bb.ask()
            X_train.append(x)
            y_train.append(y)

            self.update_prob(y)

            for col in range(len(x)):

                if (y, col) not in self.avg_matrix.keys():
                    self.avg_matrix[(y, col)], self.var_matrix[(y, col)] = x[col], float('nan')
                    self.old_var[(y, col)] = x[col]
                else:
                    self.avg_matrix[(y, col)], self.var_matrix[(y, col)] = \
                        self.update_param(self.avg_matrix[(y, col)], self.var_matrix[(y, col)], x[col], y, col)

            if (i + 1) % 10 == 0:
                obj = Classifier(self.p_label, self.avg_matrix, self.var_matrix)
                acc = obj.predict(X_test, y_test)
                accuracy.append(acc)

        # ---------------------------------------

        # Plot accuracy and write submission
        self.output(accuracy)

    # -------------------------------------------------------

    def update_param(self, cur_avg, cur_var, x, y, col):
        """
        Return updated avg and variance
        """
        """
        Calculating/updating running Mean
        https://math.stackexchange.com/questions/106700/incremental-averageing
        u: avg, n: num of obvs: u(n) = u(n-1) + [x(nth) - u(n-1) / n]
        """
        new_avg = cur_avg + ((x - cur_avg)/self.label_count[y])

        """
        Calculating/updating running Variance
        Ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        Sample var(n) = var(n) + ((x(n) - u(n-1)*(x(n) - u(n)) - var(n))/(n-1)
        Division by (n-1) and not n due to Bessel's Correction, or Sample Variance
        """
        if np.isnan(cur_var):
            new_var = np.var([x, self.old_var[(y, col)]], ddof=1)  # ddof for sample variance
        else:
            new_var = cur_var - (cur_var/(self.label_count[y]-1)) + (((x - cur_avg)**2)/(self.label_count[y]-1))

        return new_avg, new_var

    # -------------------------------------------------------

    def update_prob(self, y):
        """
        Calculate Probability of label
        :param y: input lable seen
        :return:
        """

        if y in self.labels:
            self.label_count[y] += 1
        else:
            self.label_count[y] = 1

        self.labels = list(self.label_count.keys())

        for label in self.labels:
            self.p_label[label] = self.label_count[label]/sum(self.label_count.values())

    # -------------------------------------------------------

    def output(self, accuracy):
        """
        Make Plot and Write Submission File
        :param accuracy: List of accuracies for 10th iteration
        :return:
        """

        iterations = [i for i in range(10, self.train_size + 1, 10)]

        submission = pd.DataFrame()
        submission[0] = iterations
        submission[1] = accuracy
        submission.to_csv("results_" + self.name + ".txt", index=False, header=False, sep=' ')

# -------------------------------------------------------


class Classifier:

    def __init__(self, p_label, avg_matrix, var_matrix):

        self.p_label = p_label
        self.avg_matrix = avg_matrix
        self.var_matrix = var_matrix
        self.labels = sorted(self.p_label.keys())

    # -------------------------------------------------------

    def predict(self, X_test, y_test):

        y_pred = []  # predicted labels array

        for i in range(len(X_test)):  # For each value/row of X_test

            # Initialize dictionary to store pdf/label
            res: dict = {label: 0 for label in self.labels}

            for label in self.labels:  # Calculate probabilities of each label

                # For each attrib as P(x1/Y)*P(x2/Y)...P(xn/Y) for each column
                for col in range(len(X_test.columns)):

                    # P(y/x1,...xn) = P(x1/y)*...P(xn/y)
                    # Can use log to avoid numerical underflow (log(P(x1/y)) + ... + log(P(xn/y))
                    pdf = self.gaussian(X_test.iloc[i, col], self.avg_matrix[(label, col)],
                                        self.var_matrix[(label, col)])

                    if pdf == 0.0 or np.isnan(pdf):
                        res[label] += 0
                    else:
                        res[label] += np.log(pdf)

                # P(Y), as P(y/x1,...xn) = P(x1/y)*...P(xn/y) * P(Y)
                if label not in self.labels:
                    prob_y = 0.0
                else:
                    prob_y = self.p_label[label]

                if prob_y == 0.0:
                    res[label] += 0
                else:
                    res[label] += np.log(prob_y)

                # Zero out nan values, to help choose argmax
                if np.isnan(res[label]):
                    res[label] = 0.0

            # Convert res to a int:float dict
            res1 = {int(k): float(v) for k, v in res.items()}
            v = list(res1.values())
            k = list(res1.keys())
            # Store the label against max probability of labels for each test row/value
            # Essentially the argmax of P(Y/x)
            y_pred.append(k[v.index(max(v))])

        return self.acc_store(y_test, y_pred)

    # -------------------------------------------------------

    def gaussian(self, x, avg, var):

        """
        Gaussian Distribution, p(x) = 1/sqrt(2*pi*var) * e^(-1/2*square(x-avg)/var) * dx
        For probability density function, we don't multiply by dx, pdf(x) = P(x/Y)
        probability cannot be more than 1, but pdf w/o dx can be > 1 (even infinite)

        :param x: Input Feature/Lable
        :param avg: mean
        :param var: variance
        :return: probability distribution function
        """

        if var == 0.0 or np.isnan(var):
            return 0.0
        dnm = (2 * np.pi * var) ** .5
        num = np.exp(-(float(x) - float(avg)) ** 2 / (2 * var))
        pdf = num/dnm

        return pdf

    # -------------------------------------------------------

    def acc_store(self, y_test, y_pred):
        """
        Calculate correctly classified labels
        :param y_test: Actual Labels
        :param y_pred: Predicted Labels
        :return: accuracy
        """

        # Calculate correctly classified labels
        correct = 0
        for i in range(len(y_pred)):
            if y_test[i] == y_pred[i]:
                correct += 1

        return round(correct / len(y_pred), 3)


# -------------------------------------------------------


if __name__ == '__main__':
    obj = StoreData()
    obj.input()

