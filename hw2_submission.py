import collections
import math
import numpy as np
import re
import string


class Logistic_Regression():
    def __init__(self):
        self.W = None
        self.b = None
    
        
    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-2):
        """
        Train this Logistic Regression classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """  
        dim = x.shape[1]
        num_train = x.shape[0]

        #initialize W
        if self.W == None:
            self.W = 0.001 * np.random.randn(dim, 1)
            self.b = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and update W, b 
            #cross- entropy loss. around 30 min in feb 11 lecture
            z = x_batch.dot(self.W) + self.b
            z = np.clip(z, -500, 500)
            y_pred = self.sigmoid(z)
            #y_pred = y_batch[0] #CHANGED FROM X_BATCH TO Y_BATCH
            for i in range(len(y_pred)):
                if y_pred[i] <.5:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            loss, gradient = self.loss(x_batch, y_pred, y_batch)
            for i in range(len(self.W)):
                self.W[i] = self.W[i] - (learning_rate * gradient['dW'][i])
            self.b = self.b - (learning_rate * gradient['db'])
            acc = (np.mean(y_pred == y_batch))
            #print("PRINTING ACC:")
            #print(acc)
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################
            
            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y
        #linear function wx+b
        print("shape of x =" + str(x.shape))
        print("shape of self.w =" + str(self.W.shape))
        z = x.dot(self.W)#self.W.dot(x) + self.b#x.dot(np.transpose(self.W)) + self.b
        z = np.clip(z, -500, 500)
        y_pred = self.sigmoid(z)
        for i in range(len(y_pred)):
            if y_pred[i] <.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_pred

    
    def loss(self, x_batch, y_pred, y_batch):
        """
        Compute the loss function and its derivative. 
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient dictionary with two keys : 'dW' and 'db'
        """
        gradient = {'dW' : None, 'db' : None}
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate loss and gradient
        #cross-entropy loss
        loss = 0
        
        for i in range(len(y_pred)):
            loss += y_batch[i]*math.log(np.where(y_pred[i] == 0, 1e-5, y_pred[i]),10) + (1-y_batch[i])*math.log(np.where(1-y_pred[i] == 0, 1e-5, 1-y_pred[i]),10)
        
        dW = np.transpose(x_batch).dot(y_pred - y_batch)
        #np.transpose(x_batch)*(y_pred - y_batch)#(y_pred - y_batch).dot(x_batch)#
        db = (y_pred - y_batch)
            
        loss = (-1/len(y_pred))*loss
        #dW = #TODO: dW NOW OR EARLIER?
        gradient['dW'] = dW#dW/len(y_pred)
        #DW LECTURE FEB 16 41:27 
        #print("DW = " + str(dW))
        gradient['db'] = db
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return loss, gradient

    
    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """ 
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # IGNORE Calculate loss and update W 
        #Should just be one line of code
        s = 1/(1+np.exp(-z))

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s

class Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """
        
        self.x = X_train
        self.y = y_train  
        
        self.gen_by_class()
        
    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = None
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)
        for i in range(len(self.x)):
            self.x_by_class[self.y[i]] = self.x[i]
            self.mean_by_class[self.y[i]] = self.mean(self.x[i])
            #print("TEST = " + str(self.x[i].shape))#self.mean_by_class[self.y[i]]))
            self.std_by_class[self.y[i]] = self.std(self.x[i])
            #TODO: SELF.Y_PRIOR
            #self.y = +

        pass

        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        mean = (x)/(len(x))
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean
    
    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std
        mean = self.mean(x)
        summed = 0
        #summed = (x - mean)**2
        for i in range(len(x)):
            summed += np.dot(x[i] - mean, x[i] - mean)#(x[i] - mean)**2
        std = math.sqrt(summed/len(x))
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return std
    
    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std
        first = 1/(std*math.sqrt(2*math.pi))
        second = math.exp(-.5*((x-mean)/std)**2)
        gaussian = first*second
        print("GAUSSIAN = " + str(gaussian))
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """
            
        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        print("num_class = " + str(num_class))
        p
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction

    
class Spam_Naive_Bayes(object):
    """Implementation of Naive Bayes for Spam detection."""
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)
 
    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)
 
    def get_word_counts(self, words):
        """
        Generate a dictionary 'word_counts' 
        Hint: You can use helper function self.clean and self.toeknize.
              self.tokenize(x) can generate a list of words in an email x.

        Inputs:
            -words : list of words that is used in a data sample
        Output:
            -word_counts : contains each word as a key and number of that word is used from input words.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        word_counts = dict()
        #words = self.clean(words)
        for i in range(len(words)):
            words[i] = self.clean(words[i])
            if words[i] in word_counts:
                word_counts[words[i]] += 1#TODO: Will this be a problem?
            else:
                word_counts[words[i]] = 1#TODO: Will this be a problem?
        #print("word_count =" + str(word_counts))
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
            
        return word_counts

    def fit(self, X_train, y_train):
        """
        compute likelihood of all words given a class

        Inputs:
            -X_train : list of emails
            -y_train : list of target label (spam : 1, non-spam : 0)
            
        Variables:
            -self.num_messages : dictionary contains number of data that is spam or not
            -self.word_counts : dictionary counts the number of certain word in class 'spam' and 'ham'.
            -self.class_priors : dictionary of prior probability of class 'spam' and 'ham'.
        Output:
            None
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        num_messages = {}
        num_messages['spam'] = 0
        num_messages['ham'] = 0
        
        word_counts = {}
        word_counts['spam'] = {}
        word_counts['ham'] = {}
        class_priors = {}
        
        for i in range(len(y_train)):
            if y_train[i] == 1:
                num_messages['spam'] += 1
            else:
                num_messages['ham'] += 1
        total = num_messages['spam'] + num_messages['ham']
        class_priors['spam'] = num_messages['spam']/total
        class_priors['ham'] = num_messages['ham']/total
        
        for x, y in zip(X_train, y_train):
            count_dict = self.get_word_counts(self.tokenize(x))
            for word in count_dict:
                if y == 1:
                    if word in word_counts['spam']:
                        word_counts['spam'][word] += 1
                    else:
                        word_counts['spam'][word] = 1
                elif y == 0:
                    if word in word_counts['ham']:
                        word_counts['ham'][word] += 1
                    else:
                        word_counts['ham'][word] = 1
        self.num_messages = num_messages
        self.word_counts = word_counts
        self.class_priors= class_priors
        
        
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
                
    def predict(self, X):
        """
        predict that input X is spam of not. 
        Given a set of words {x_i}, for x_i in an email(x), if the likelihood 
        
        p(x_0|spam) * p(x_1|spam) * ... * p(x_n|spam) * y(spam) > p(x_0|ham) * p(x_1|ham) * ... * p(x_n|ham) * y(ham),
        
        then, the email would be spam.

        Inputs:
            -X : list of emails

        Output:
            -result : A numpy array of shape (N,). It should tell rather a mail is spam(1) or not(0).
        """
            
        result = []
        #print("word_counts=")
        #print(self.word_counts['spam'])
        for x in X:
            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # calculate naive bayes probability of each class of input x
            num_spam = self.num_messages['spam']
            num_ham = self.num_messages['ham']
            words = self.get_word_counts(self.tokenize(x))
            spam_prob = 0
            ham_prob = 0
            for word in words:
                if word in self.word_counts['spam']:
                    spam_prob += math.log(self.word_counts['spam'][word]/num_spam,10)
                else:
                    spam_prob += math.log(0.0000001,10)
                if word in self.word_counts['ham']:
                    ham_prob += math.log(self.word_counts['ham'][word]/num_ham,10)
                else:
                    ham_prob += math.log(0.0000001,10)
            if (spam_prob > ham_prob):
                result.append(1)
            else:
                result.append(0)
            pass
            # END_YOUR_CODE
            ############################################################
            ############################################################
        #print("result=")
        #print(result)  
        result = np.array(result)
        print("len(result) =" +str(len(result)))
        print("len(X) =" + str(len(X)))
        return result
    