import collections
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB

class Gaussian_Naive_Bayes():
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
        self.y_prior = []#None
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)
        testing = GaussianNB()
        #print("TEST=")
        #print(testing.fit(self.x,self.y))
        #print(testing.get_params(deep=True))
        #print(vars(testing))#["sigma_"])
        for i in range(len(self.x)):
            if self.y[i] in self.x_by_class:
                len_thing = np.shape(self.x_by_class[self.y[i]])
                if len(len_thing) == 1:
                    self.x_by_class[self.y[i]] = np.append([self.x_by_class[self.y[i]]], [self.x[i]], axis=0)
                else:
                    self.x_by_class[self.y[i]] = np.append(self.x_by_class[self.y[i]], [self.x[i]], axis=0)
            else:
                self.x_by_class[self.y[i]] = self.x[i]
                
        for i in range(len(self.x)):
            self.mean_by_class[self.y[i]] = self.mean(self.x_by_class[self.y[i]])
            self.std_by_class[self.y[i]] = self.std(self.x_by_class[self.y[i]])
            
        for key in np.unique(self.y):
            self.y_prior = np.append(self.y_prior, len(self.x_by_class[key])/len(self.x))
        #print("Prior = ")
        #print(self.y_prior)
        #print("mean[0] = ")
        #print(self.mean_by_class[0])
        #print("std = ")
        #print(self.std_by_class)

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        mean = 0
        for i in range(len(x)):
            mean += x[i]
        mean = mean/len(x)
        pass;
    
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
        for i in range(len(x)):
            summed += np.square(x[i] - mean)
        std = np.sqrt((summed/(len(x)-1))+1e-9)#TODO: Choose -1 or not
        for i in range(len(std)):
            if std[i] == 0:
                std[i] = 0.00001
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
        #print("std =")
        #print(std)
        #print("thing=")
        #print(np.sqrt(2*math.pi))
        first = 1/(std*np.sqrt(2*math.pi))
        #print("first =")
        #print(first)
        blue = np.square((x-mean)/std)
        #print("blue = ")
        #print(blue)
        second = np.exp(-.5*blue)
        #print("second =")
        #print(second)
        gaussian = first*second
        #print("GAUSSIAN = ")
        #print(gaussian)
        for i in range(len(gaussian)):
            if gaussian[i] == 0:
                gaussian[i] = 1e-9
        #print("GAUSSIAN = ")
        #print(gaussian)
        pass;
        # END_YOUR_CODE
        return gaussian
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
        #print("x[0] = ")
        #print(x[0])
        #prior = [0.36220472, 0.63779528]
        for i in range(n):
            for key in range(num_class):
                #print("AH =")
                #print(np.sum(np.log(self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])*prior[key])))#np.log(self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])))
                #print("key =")
                #print(key)
                #print("Boop=" + str(booper))
                #print(self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])*self.y_prior[key])
                #np.where(y_pred[i] == 0, 1e-5, y_pred[i])
                helper = self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])#*self.y_prior[key]
                prediction[i][key] = np.sum(np.where(np.log(helper)==0, 1e-5+self.y_prior[key], np.log(helper)+self.y_prior[key]))
                #self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])*prior[key]
#        for i in range(n):
#            test = 0
#            for key, value in self.x_by_class.items():
#                x_probs = self.calc_gaussian_dist(x[i], self.mean_by_class[key], self.std_by_class[key])
#                if (x_probs > test):
#                    test = x_probs
#                    prediction[i] = key
        pass;
    
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction


class Neural_Network():
    def __init__(self, hidden_size = 64, output_size = 1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None    
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
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
        if self.W1 == None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0
            
            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]
            
            loss, gradient = self.loss(x_batch, y_batch)
            ##
            #print("gradient =")
            #print(gradient)
            #print("loss = ")
            #print(loss)
            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method
            self.W1 = self.W1 - (learning_rate*gradient['dW1'])
            self.W2 = self.W2 - (learning_rate*gradient['dW2'])
            self.b1 = self.b1 - (learning_rate*gradient['db1'])
            self.b2 = self.b2 - (learning_rate*gradient['db2'])
            pass;
        
            # END_YOUR_CODE
            ############################################################
            ############################################################
            
            y_pred = self.predict(x_batch)
            ###
            #print("y_pred = ")
            #print(y_pred)
            #print("y_batch =")
            #print(y_batch)
            ###
            acc = np.mean(y_pred == y_batch)
            
            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))
                
    def loss(self, x_batch, y_batch, reg = 1e-3):
            """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives. 

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer.

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
            gradient = {'dW1' : None, 'db1' : None, 'dW2' : None, 'db2' : None}


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate y_hat which is probability of the instance is y = 0.
            #print("TESTING W1[0]=")
            #print(self.W1[0])
            #print("TESTING x-batch.T[0]=")
            #print(x_batch.T[0])
            #g1 = np.zeros((64,1))

            #for sa in range(len(x_batch.T)):
            #    g1 += np.multiply(x_batch.T[sa].reshape((64,1)),(self.W1[sa]).reshape((64,1)))
            #print("g1 PRIOR=")
            #print(g1)
            #g1 += self.b1
            #print("self.b1 =")
            #print(self.b1)
            g1 = x_batch.dot(self.W1) + self.b1#self.W1.T *x_batch + self.b1#
            #print("g1 =")
            #g1 = (self.W1.T * x_batch) + self.b1
            #print(g1.shape)
            #print(g1)

            h1 = self.activation(g1)

            #print("w2 =")
            #print(self.W2.T.shape)
            #print(self.W2)
            #print("h1 = ")
            #print(h1.shape)
            #print(h1)

            #g2 = 0
            #for sb in range(len(self.W2)):
            #    g2 += np.multiply(self.W2[sb], h1[sb]) + self.b2
            #print("b2 =")
            #print(self.b2)
            #print("self.W2 =")
            #print(self.W2)
            #print("h1 =")
            #print(h1)
            #print("self.b2=")
            #print(self.b2)
            #g2 = np.multiply(self.W2, h1) + self.b2
            g2 = h1.dot(self.W2) + self.b2#(self.W2) + self.b2 #h1.dot(self.W2) + self.b2#(self.W2.T * h1) + self.b2#h1.dot(self.W2) + self.b2#h1.T.dot(self.W2)+self.b2#self.W2.T.dot(h1) + self.b2#
            #print("g2 =")
            #print(g2.shape)
            #print(g2)
            
            y_hat = self.sigmoid(g2)
            #print("y_hat PRIOR=")
            #print(y_hat.shape)
            #print(y_hat)
            y_hat2 = y_hat.copy()
            for i in range(len(y_hat)):                    
                if y_hat[i] <.5:
                    y_hat2[i] = 0 + 1e-5
                else:
                    y_hat2[i] = 1 - 1e-5
            #print("y_hat2 =")
            #print(y_hat.shape)
            #print(y_hat2)
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and gradient
            summed = 0
            for i in range(len(y_batch)):
                summed += y_batch[i]*np.log(y_hat2[i]) + (1-y_batch[i])*np.log(1-y_hat2[i])
            loss = (-1/len(y_batch))*summed
            #print("FABRIEN NORM W1=")
            #print(np.linalg.norm(self.W1))
            W1pow = pow(np.linalg.norm(self.W1),2)
            W2pow = pow(np.linalg.norm(self.W2),2)
            b1pow = pow(np.linalg.norm(self.b1),2)
            b2pow = pow(np.linalg.norm(self.b2),2)
            loss += reg*(W1pow + W2pow + b1pow + b2pow)
            
            #print("loss = ")
            #print(loss)
            dL_dy = (y_hat - y_batch)/np.where(y_hat*(1-y_hat)==0, 1e-5, y_hat*(1-y_hat))#SAME AS OTHER CALCULATION FOR THE DL_DY
            dy_dg2 = y_hat*(1-y_hat)
            dg2_dh1 = np.transpose(self.W2)
            dh1_dg1 = g1.copy()
            for j in range(len(g1)):
                for k in range(len(g1[j])):
                    if g1[j][k] >= 0:
                        dh1_dg1[j][k] = 1
                    else:
                        dh1_dg1[j][k] = 0#1 #0
            dg1_dW1 = np.transpose(x_batch)
            #dL_dW1 = np.dot(dg1_dW1,(dL_dy * dy_dg2 * dg2_dh1 * dh1_dg1))#dL_dy * dy_dg2 * dg2_dh1 * dh1_dg1 * dg1_dW1
            red1 = x_batch.T.dot(y_hat-y_batch)
            
            #sum_W2 = self.W2.copy()
            #for z in range(len(sum_W2)):
            #    sum_W2[z] = 0
            #for q in range(len(dh1_dg1[0])):
            #    useful = dh1_dg1[q].reshape((len(dh1_dg1[0]),1))
            #    sum_W2 += np.multiply(self.W2, useful)
            #print("sum_W2=")
            #print(sum_W2.shape)#(64,1)
            #sum_W2 = np.multiply(self.W2, h1)
            #sum_W2 = np.dot(h1, self.W2)
            #print("sum_W2 shape=")
            #print(sum_W2.shape)
            #red1 = x_batch.T.dot(y_hat-y_batch)#(30,1)
            #red2 = red1.dot(sum_W2.T)#(30,64)
            #red3 = red2 + (2*reg*self.W1)

            #dL_dW1 = red2 + (2*reg*self.W1)
            W1a = (y_hat-y_batch)
            W1b = W1a.dot(self.W2.T)
            W1c = np.multiply(W1b, dh1_dg1)
            dL_dW1 = ((1/len(y_batch))*x_batch.T.dot(W1c)) + (2*reg*self.W1)
            #print("size of dL_dw1 =")
            #print(dL_dW1.shape)

            #dL_dW1 = np.dot(x_batch.T , np.dot(y_hat - y_batch, sum_W2.T)) + (2*self.W1*reg)#(y_hat - y_batch) * (np.multiply(self.W2, dh1_dg1)).T + (2*self.W1*reg)
            
            dg1_db1 = 1

            #blue1 = dL_dy.T.dot(dy_dg2)
            #blue2 = blue1.dot(dg2_dh1)
            #blue3 = blue2.dot(dh1_dg1)
            #print(blue3)
            #print("size of blue2=")
            #print(blue2.shape)
            #print("size of dh1_dg1=")
            #print(dh1_dg1.shape)
            ##dL_db1 = float(blue2.dot(dh1_dg1))
            #dL_db1 = np.dot(dh1_dg1, dL_dy)
            #dL_db1 = dL_db1 * dy_dg2
            #dL_db1 = np.dot(dg2_dh1, dL_db1)
            #dL_db1 = float(dL_db1 * dg1_db1)
            b1a = (y_hat-y_batch)
            b1b = b1a.dot(self.W2.T)
            b1c = np.multiply(b1b, dh1_dg1)
            #print("LEN(B1C)=")
            #print(len(b1c))
            b1d = np.zeros((1,len(b1c[0])))#(1xp)
            #print("SIZE OF B1D PRIOR=")
            #print(b1d.shape)
            for ba in range(len(b1c)):
                b1d += b1c[ba]
            b1d = b1d/len(b1c)
            #print("SIZE OF B1D AFTER=")
            #print(b1d.shape)
            dL_db1 = b1d + (2*reg*self.b1)

            dg2_dW2 = np.transpose(dh1_dg1)#h1)

            dL_dW2 = (1/len(y_batch))*(np.dot(h1.T,y_hat-y_batch)) + (2*reg*self.W2)
            #dL_dW2 = np.dot(dg2_dW2, (dL_dy * dy_dg2))#dL_dy * dy_dg2 * dg2_dW2
            #dL_dW2 = np.dot(h1, y_hat - y_batch) + (2*self.W2*reg)#np.multiply(y_hat - y_batch,h1) + (2*self.W2*reg)#np.multiply(h1, y_hat - y_batch) + (2*self.W2*reg)
            #np.dot(h1, y_hat - y_batch) + (2*self.W2*reg)
            
            dg2_db2 = 1
            #dL_db2 = float(np.dot(dL_dy.T, dy_dg2))#float(np.dot(dy_dg2.T,dL_dy) * dg2_db2)
            y_mean = y_batch-y_hat
            b2a = 0
            for bb in range(len(y_batch)):
                b2a += y_mean[bb]
            dL_db2 = float((b2a/len(y_batch)) + (2*reg*self.b2))

            gradient['dW1'] = dL_dW1.copy()
            gradient['db1'] = dL_db1.copy()
            gradient['dW2'] = dL_dW2.copy()
            gradient['db2'] = dL_db2
            
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################
            return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """ 
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU 
        s = z.copy()
        #print("z[0] = ")
        #print(z[0])
        if(len(z) == 1):
            if z >= 0:
                s=z
            else:
                s = 0
        elif(len(z[0]) > 1):
            for i in range(len(z)):
                for j in range(len(z[i])):
                    if z[i][j] >= 0:
                        s[i][j] = z[i][j]
                    else:
                        s[i][j] = 0
        else:
            for i in range(len(z)):
                if z[i] >= 0:
                    s[i] = z[i]
                else:
                    s[i] = 0
        #print("z[0] =")
        #print(z[0])
        #print("z =")
        #print(z)
        #print("s =")
        #print(s)
        pass;
    
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s
        
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
        #print("SIGMOID z=")
        #print(z)
        s = 1/(1+np.exp(np.clip(-z, -500, 500)))

        #print("SIGMOID s =")
        #print(s)
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s
    
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
        #print("PREDICT x =")
        #print(x.shape)
        g1 = x.dot(self.W1) + self.b1
        #g1 = np.zeros((64,1))

        #for sa in range(len(x.T)):
        #    g1 += np.multiply(x.T[sa].reshape((64,1)),(self.W1[sa]).reshape((64,1)))
        #g1 += self.b1
        h1 = self.activation(g1)
        #print("h1 =")
        #print(h1.shape)
        g2 = h1.dot(self.W2) + self.b2
        #g2 = np.multiply(self.W2, h1) + self.b2
        #g2 = g1.dot(self.W2)
        #print("g2 =")
        #print(g2.shape)
        #print(g2)
        y_hat = self.sigmoid(g2)
        #print("y_hat=")
        #print(y_hat.shape)
        for i in range(len(y_hat)):
                if y_hat[i] <.5:
                    y_hat[i] = 0
                else:
                    y_hat[i] = 1
        #print("y_hat =")
        #print(y_hat)
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_hat

    