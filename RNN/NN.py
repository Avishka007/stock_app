from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib import style
import time
import warnings
import numpy as np
from numpy import newaxis
import csv
import pandas as pd 
import ast
from numpy import diff
from statistics import mean
import loaddata


#Loading Stock Data from saved CSV.
X_train, y_train, X_test, y_test = loaddata.load_data('./stock/GOOG.csv', 55, True)
#stocks = AMZON , APPL , citigroup , dowjones , ebay , GOOG , KO , TATA , test 


class RecurrentNeuralNetwork:
    def __init__ (self, xs, ys, rl, eo, lr):
        #initial input 
        self.x = np.zeros(xs)
        #input size 
        self.xs = xs
        #expected output 
        self.y = np.zeros(ys)
        #output size
        self.ys = ys
        #weight matrix for interpreting results from LSTM cell
        self.w = np.random.random((ys, ys))
        #matrix used in RMSprop
        self.G = np.zeros_like(self.w)
        #length of the recurrent network - number of recurrences 
        self.rl = rl
        #learning rate 
        self.lr = lr
        #array for storing inputs
        self.ia = np.zeros((rl+1,xs))
        #array for storing cell states
        self.ca = np.zeros((rl+1,ys))
        #array for storing outputs
        self.oa = np.zeros((rl+1,ys))
        #array for storing hidden states
        self.ha = np.zeros((rl+1,ys))
        #forget gate 
        self.af = np.zeros((rl+1,ys))
        #input gate
        self.ai = np.zeros((rl+1,ys))
        #cell state
        self.ac = np.zeros((rl+1,ys))
        #output gate
        self.ao = np.zeros((rl+1,ys))
        #array of expected output values
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        #declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(xs, ys, rl, lr)
    
    #activation function. simple nonlinearity, convert nums into probabilities between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #the derivative of the sigmoid function. used to compute gradients for backpropagation
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))    
    
    #lets apply a series of matrix operations to our input (curr word) to compute a predicted output (next word)
    def forwardProp(self):
        for i in range(1, self.rl+1):
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            cs, hs, f, c, o = self.LSTM.forwardProp()
            #store computed cell state
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs)) #input times weight....  
            self.x = self.eo[i-1]
        return self.oa
   
    
    def backProp(self):
        #update our weight matrices (Both in our Recurrent network, as well as the weight matrices inside LSTM cell)
        #init an empty error value 
        totalError = 0
        #initialize matrices for gradient updates
        #First, these are RNN level gradients
        #cell state
        dfcs = np.zeros(self.ys)
        #hidden state,
        dfhs = np.zeros(self.ys)
        #weight matrix
        tu = np.zeros((self.ys,self.ys))
        #Next, these are LSTM level gradients
        #forget gate
        tfu = np.zeros((self.ys, self.xs+self.ys))
        #input gate
        tiu = np.zeros((self.ys, self.xs+self.ys))
        #cell unit
        tcu = np.zeros((self.ys, self.xs+self.ys))
        #output gate
        tou = np.zeros((self.ys, self.xs+self.ys))
        #loop backwards through recurrences
        for i in range(self.rl, -1, -1):
            #error = calculatedOutput - expectedOutput
            error = self.oa[i] - self.eo[i]
            #calculate update for weight matrix
            #(error * derivative of the output) * hidden state
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            #Time to propagate error back to exit of LSTM cell
            #1. error * RNN weight matrix
            error = np.dot(error, self.w)
            #2. set input values of LSTM cell for recurrence i (horizontal stack of arrays, hidden + input)
            self.LSTM.x = np.hstack((self.ha[i-1], self.ia[i]))
            #3. set cell state of LSTM cell for recurrence i (pre-updates)
            self.LSTM.cs = self.ca[i]
            #Finally, call the LSTM cell's backprop, retreive gradient updates
            #gradient updates for forget, input, cell unit, and output gates + cell states & hiddens states
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i-1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs)
            #calculate total error (not necesarry, used to measure training progress)
            totalError += np.sum(error)
            #accumulate all gradient updates
            #forget gate
            tfu += fu
            #input gate
            tiu += iu
            #cell state
            tcu += cu
            #output gate
            tou += ou
        #update LSTM matrices with average of accumulated gradient updates    
        self.LSTM.update(tfu/self.rl, tiu/self.rl, tcu/self.rl, tou/self.rl) 
        #update weight matrix with average of accumulated gradient updates  
        self.update(tu/self.rl)
        #return total error of this iteration
        return totalError
    
    def update(self, u):
        #vanilla implementation of RMSprop
        self.G = 0.9 * self.G + 0.1 * u**2  
        self.w -= self.lr/np.sqrt(self.G + 1e-8) * u
        return
    
    #this is where we generate some prediction sequence after having fully trained our model
    def preciction(self):
         #loop through recurrences - start at 1 so the 0th entry of all arrays will be an array of 0's
        for i in range(1, self.rl+1):
            #set input for LSTM cell, combination of input (previous output) and previous hidden state
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            #run forward prop on the LSTM cell, retrieve cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            #store input as vector
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x #Use np.argmax?
            #store cell states
            self.ca[i] = cs
            #store hidden state
            self.ha[i] = hs
            #forget gate
            self.af[i] = f
            #input gate
            self.ai[i] = inp
            #cell state
            self.ac[i] = c
            #output gate
            self.ao[i] = o
            #calculate output by multiplying hidden state with weight matrix
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            #compute new input
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        #return all outputs    
        return self.oa


class LSTMRNN:
    # LSTM cell (input, output, amount of recurrence, learning rate)
    def __init__ (self, xs, ys, rl, lr):
        #input is csv data points
        self.x = np.zeros(xs+ys)
        #input size is prediction window legnth(50)
        self.xs = xs + ys
        #output 
        self.y = np.zeros(ys)
        #output size
        self.ys = ys
        #cell state intialized as size of prediction
        self.cs = np.zeros(ys)
        #how often to perform recurrence
        self.rl = rl
        #balance the rate of training (learning rate)
        self.lr = lr
        #init weight matrices for our gates
        #forget gate
        self.f = np.random.random((ys, xs+ys))
        #input gate
        self.i = np.random.random((ys, xs+ys))
        #cell state
        self.c = np.random.random((ys, xs+ys))
        #output gate
        self.o = np.random.random((ys, xs+ys))
        #forget gate gradient
        self.Gf = np.zeros_like(self.f)
        #input gate gradient
        self.Gi = np.zeros_like(self.i)
        #cell state gradient
        self.Gc = np.zeros_like(self.c)
        #output gate gradient
        self.Go = np.zeros_like(self.o)
    
    #activation function to activate our forward prop, just like in any type of neural network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #derivative of sigmoid to help computes gradients
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    #tanh! another activation function, often used in LSTM cells
    #Having stronger gradients: since data is centered around 0, 
    #the derivatives are higher. To see this, calculate the derivative 
    #of the tanh function and notice that input values are in the range [0,1].
    def tangent(self, x):
        return np.tanh(x)
    
    #derivative for computing gradients
    def dtangent(self, x):
        return 1 - np.tanh(x)**2
    
    #lets compute a series of matrix multiplications to convert our input into our output
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o
    
    #now calculete error derivatives
    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        #error = error + hidden state derivative. clip the value between -6 and 6.
        e = np.clip(e + dfhs, -6, 6)
        #multiply error by activated cell state to compute output derivative
        do = self.tangent(self.cs) * e
        #output update = (output deriv * activated output) * input
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        #derivative of cell state = error * output * deriv of cell state + deriv cell
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        #deriv of cell = deriv cell state * input
        dc = dcs * i
        #cell update = deriv cell * activated cell * input
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        #deriv of input = deriv cell state * cell
        di = dcs * c
        #input update = (deriv input * activated input) * input
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        #deriv forget = deriv cell state * all cell states
        df = dcs * pcs
        #forget update = (deriv forget * deriv forget) * input
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        #deriv cell state = deriv cell state * forget
        dpcs = dcs * f
        #deriv hidden state = (deriv cell * cell) * output + deriv output * output * output deriv input * input * output + deriv forget
        #* forget * output
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df, self.f)[:self.ys] 
        #return update gradinets for forget, input, cell, output, cell state, hidden state
        return fu, iu, cu, ou, dpcs, dphs
           
    def update(self, fu, iu, cu, ou):#this function is the core of LSTM this updates and maintains cell state
        #update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2 
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2   
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2   
        self.Go = 0.9 * self.Go + 0.1 * ou**2   
        
        #update our gates using our gradients
        self.f -= self.lr/np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr/np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr/np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr/np.sqrt(self.Go + 1e-8) * ou
        return


model = Sequential()
model.add(LSTM(
    input_dim=1,
    output_dim=50,  
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')

model.fit(
    X_train,
    y_train,
    #initialize batch size here:
    batch_size=512,
    #initialize epoch here:
    nb_epoch=3,
    validation_split=0.05)


style.use('ggplot')


def many_predictions(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    #to highlite the prediction time in the future/ the pink bar.
    p = plt.axvspan(411,434, facecolor='#db5978', alpha=0.5,label= 'future prediction window')
    #print(predicted_data)
    ax.plot(true_data, label='Real Data')
    print ('process done open the plot')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data[1:]):
        padding = [None for p in range((i+1) * prediction_len)]
        plt.plot(padding + data,color='#2920b2')
        plt.xlabel('Years')
        plt.ylabel('Closing price')
        x = np.array([0,100,200,300,400,411,434])
        my_xticks = ['2000-JAN','2004-JUL','2009-JAN','2013-JUL','2017-JAN','2018-JAN','2018-DEC']
        plt.xticks(rotation=45)
        plt.xticks(x, my_xticks)
        plt.legend()
    plt.title("closing prices \n From 2000-january-1st to 2018-january-1st ")

    ##################################################################################################
    rr=[]
    rr=predicted_data

    #ss is the numpy array of predicted data set. there 7 predictions.
    ss=(predicted_data)

    #a = np.asarray(ss)
    #np.savetxt("predicted_data.csv", a, delimiter=",")

    #save ss array to a csv:
    df = pd.DataFrame(ss)
    df.to_csv("p_values.csv")
    print("p_values.csv has been updated")

    
#open the saved csv and plot all 50 records to a new prediction plot
    file = open('p_values.csv',newline='')
    reader = csv.reader(file)

    header = next(reader)

    data=[]

    for row in reader:
        p1=(row[1])
        p2=(row[2])
        p3=(row[3])
        p4=(row[4])
        p5=(row[5])
        p6=(row[6])
        p7=(row[7])
        p8=(row[8])
        p9=(row[9])
        p10=(row[10])
        p11=(row[11])
        p12=(row[12])
        p13=(row[13])
        p14=(row[14])
        p15=(row[15])
        p16=(row[16])
        p17=(row[17])
        p18=(row[18])
        p19=(row[19])
        p20=(row[20])
        p21=(row[21])
        p22=(row[22])
        p23=(row[23])
        p24=(row[24])
        p25=(row[25])
        p26=(row[26])
        p27=(row[27])
        p28=(row[28])
        p29=(row[29])
        p30=(row[30])
        p31=(row[31])
        p32=(row[32])
        p33=(row[33])
        p34=(row[34])
        p35=(row[35])
        p36=(row[36])
        p37=(row[37])
        p38=(row[38])
        p39=(row[39])
        p40=(row[40])
        p41=(row[41])
        p42=(row[42])
        p43=(row[43])
        p44=(row[44])
        p45=(row[45])
        p46=(row[46])
        p47=(row[47])
        p48=(row[48])
        p49=(row[49])
        p50=(row[50])
    #print(p1)

    s=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,p50]

    #save gathered 8th prediction plots to slope.csv to calculate the slope later
    df = pd.DataFrame(s)
    df.to_csv("slope.csv")
    print("slope.csv has been updated")

    #print(s)


    x,y=[],[]
    #read slope.csv to calculate the last predictions slope:
    with open ('slope.csv','r') as csvfile:
        plots = csv.reader(csvfile , delimiter=',')
        for row in plots:
            x.append(row[0]) 
            y.append(float(row[1])*10000)

    #print(x)
    #print(y)
    st=y[1]


    xs = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50])
    ys = np.array([y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9],y[10],y[11],y[12],y[13],y[14],y[15],y[16],y[17],y[18],y[19],y[20],y[21],y[22],y[23],y[24],y[25],y[26],y[27],y[28],y[29],y[30],y[31],y[32],y[33],y[34],y[35],y[36],y[37],y[38],y[39],y[40],y[41],y[42],y[43],y[44],y[45],y[46],y[47],y[48],y[49],y[50]      ])

    #this function is to calculate the slope of the prediction (the best fit slope):
    def best_fit_slope(xs,ys):
        m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
             ((mean(xs)* (mean(xs)) - mean(xs**2))))
        return m

    m = best_fit_slope(xs,ys)
    
    #best fit slope is y=mx+c type of function.
    #m is the slope
    #depending on the slope we can say the confidense of the prediction

    # round up the slope value
    m=round(m, 2)
    print(m,": is the value of slope")
    print(m)
    
    plt.show()

#################################################################################

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

##################################################################################

def build_model_twoLayer(layers):
    model = Sequential()
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    return model

##########################################################################################

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    prediction_len = int(prediction_len)
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


#Plotting the predictions
predictions = predict_sequences_multiple(model, X_test, 55, 50)
many_predictions(predictions, y_test, 55)



