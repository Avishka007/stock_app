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
import model



#Loading Stock Data from saved CSV.
X_train, y_train, X_test, y_test = model.load_data('./stock/GOOG.csv', 55, True)
#stocks = AMZON , APPL , citigroup , dowjones , ebay , GOOG , KO , TATA , test 


def bin2int(bin_list):
    #bin_list = [0, 0, 0, 1]
    int_val = ""
    for k in bin_list:
        int_val += str(int(k))
    #int_val = 11011011    
    return int(int_val, 2)
##########
s = [0.841470985, 0.873736397, 0.90255357, 0.927808777, 0.949402346, 0.967249058, 0.98127848, 0.991435244, 0.997679266, 0.999985904, 0.998346054, 0.992766189, 0.983268329, 0.969889958, 0.952683874, 0.931717983, 0.907075026, 0.878852258, 0.847161063, 0.812126509, 0.773886863, 0.73259304, 0.688408006, 0.64150614, 0.592072543, 0.540302306, 0.486399742, 0.430577581, 0.373056127, 0.314062391, 0.253829194, 0.192594249, 0.130599223, 0.068088781, 0.005309624, -0.057490488, -0.120063711, -0.182163097, -0.243543569, -0.303962886, -0.3631826, -0.420968998, -0.477094024, -0.531336178, -0.583481391, -0.633323869, -0.680666907, -0.725323664, -0.7671179, -0.805884672, -0.841470985, -0.873736397, -0.90255357, -0.927808777, -0.949402346, -0.967249058, -0.98127848, -0.991435244, -0.997679266, -0.999985904, -0.998346054, -0.992766189, -0.983268329, -0.969889958, -0.952683874, -0.931717983, -0.907075026, -0.878852258, -0.847161063, -0.812126509, -0.773886863, -0.73259304, -0.688408006, -0.64150614, -0.592072543, -0.540302306, -0.486399742, -0.430577581, -0.373056127, -0.314062391, -0.253829194, -0.192594249, -0.130599223, -0.068088781, -0.005309624, 0.057490488, 0.120063711, 0.182163097, 0.243543569, 0.303962886, 0.3631826, 0.420968998, 0.477094024, 0.531336178, 0.583481391, 0.633323869, 0.680666907, 0.725323664, 0.7671179, 0.805884672, 0.841470985, 0.873736397, 0.90255357, 0.927808777, 0.949402346, 0.967249058, 0.98127848, 0.991435244, 0.997679266, 0.999985904, 0.998346054, 0.992766189, 0.983268329, 0.969889958, 0.952683874, 0.931717983, 0.907075026, 0.878852258, 0.847161063, 0.812126509, 0.773886863, 0.73259304, 0.688408006, 0.64150614, 0.592072543, 0.540302306, 0.486399742, 0.430577581, 0.373056127, 0.314062391, 0.253829194, 0.192594249, 0.130599223, 0.068088781, 0.005309624, -0.057490488, -0.120063711, -0.182163097, -0.243543569, -0.303962886, -0.3631826, -0.420968998, -0.477094024, -0.531336178, -0.583481391, -0.633323869, -0.680666907, -0.725323664, -0.7671179, -0.805884672, -0.841470985, -0.873736397, -0.90255357, -0.927808777, -0.949402346, -0.967249058, -0.98127848, -0.991435244, -0.997679266, -0.999985904, -0.998346054, -0.992766189, -0.983268329, -0.969889958, -0.952683874, -0.931717983, -0.907075026, -0.878852258, -0.847161063, -0.812126509, -0.773886863, -0.73259304]

my_list = [float(i) for i in s]
r =my_list

def dataset(num):
    # num - no of samples
    bin_len = 20
    X = np.zeros((num, bin_len))
    Y = np.zeros((num))

    for i in range(num):
        X[i] = np.around(np.random.rand(bin_len)).astype(int)
        Y[i] = bin2int(X[i])
    return X, Y

no_of_smaples = 20

train_X ,train_Y = dataset(no_of_smaples)
#print (train_X)
#print (train_Y)


trainX, trainY = dataset(no_of_smaples)
testX, testY = dataset(50) # this nu is number of answers

class RNN:
    def __init__(self):
        self.W = [1, 1]
        self.W_delta = [0.001, 0.001]
        self.W_sign = [0, 0]

        self.eta_p = 1.2
        self.eta_n = 0.5

    def state(self, xk, sk):
        return xk * self.W[0] + sk * self.W[1]

    def forward_states(self, X):
        S = np.zeros((X.shape[0], X.shape[1]+1))
        for k in range(0, X.shape[1]):
            next_state = self.state(X[:,k], S[:,k])
            S[:,k+1] = next_state
        return S

    def output_gradient(self, guess, real):
        return 2 * (guess - real) / no_of_smaples

    def backward_gradient(self, X, S, grad_out):
        grad_over_time = np.zeros(( X.shape[0], X.shape[1]+1 ))
        grad_over_time[:,-1] = grad_out

        wx_grad = 0
        wr_grad = 0
        for k in range(X.shape[1], 0, -1):
            wx_grad += np.sum( grad_over_time[:, k] * X[:, k-1] )
            wr_grad += np.sum( grad_over_time[:, k] * S[:, k-1] )

            grad_over_time[:, k-1] = grad_over_time[:, k] * self.W[1]
        return (wx_grad, wr_grad), grad_over_time

    def update_rprop(self, X, Y, W_prev_sign, W_delta):
        S = self.forward_states(X)
        grad_out =  self.output_gradient(S[:, -1], Y)
        W_grads, _ = self.backward_gradient(X, S, grad_out)
        self.W_sign = np.sign(W_grads)
        print ("loss : {:0.08f}".format(np.mean(grad_out)))

        for i, _ in enumerate(self.W):
            if self.W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= self.eta_p
            else:
                W_delta[i] *= self.eta_n
        self.W_delta = W_delta

    def train(self, X, Y, training_epochs):
        for epochs in range(training_epochs):
            self.update_rprop(X, Y, self.W_sign, self.W_delta)

            for i, _ in enumerate(self.W):
                self.W[i] -= self.W_sign[i] * self.W_delta[i]


rnn = RNN()
rnn.train(trainX, trainY, 20000)
print ("Weight: \t", rnn.W)
print ("input: \t\t", testY)

y = rnn.forward_states(testX)[:, -1]
print ("Predicted: \t",y)



#creating the model
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


#traning the model
model.fit(
    X_train,
    y_train,
    #initialize batch size here:
    batch_size=512,
    #initialize epoch here:
    nb_epoch=1,
    validation_split=0.05)



style.use('ggplot')
#to hide numpy warnings
warnings.filterwarnings("ignore")

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

    #ss is the numpy array of predicted data set. there 8 predictions.
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
    
############################################################
    plt.show()
    ##########
    #
    #######
    ######
#################################################################

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  
    #print (x_train)
    #print(x_test)
    return [x_train, y_train, x_test, y_test]

#################################################################################

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

##################################################################################

#function of double layer LSTM model

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
    #Predict sequence of 50 steps before shifting prediction run forward by 55 steps
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



