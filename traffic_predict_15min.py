import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from sklearn import preprocessing
from sklearn import metrics

def loadDataSet():
    file2 = file("all_surround_cells.csv","rb")
    reader2 = csv.reader(file2)
    surround_cells = []
    for line in reader2:
        surround_cells.append(str(round(float(line[1]),5))+'_'+str(round(float(line[0]),5)))

    traffic_dataFrame = pd.read_csv("/home/johnson/tensorflow/row Data/nj06downbsloc15min_new.csv")

    MyTrafficFrame = traffic_dataFrame.reindex(columns=surround_cells)        #[2861 rows * 39 columns]

    ##miss data operation
    MyTrafficFrame =  MyTrafficFrame.dropna(axis = 1,thresh = 2820)           #[2861 rows * 19 columns]
    MyTrafficFrame = MyTrafficFrame.interpolate()                             #interpolate values
                                     
    return MyTrafficFrame,MyTrafficFrame['32.05278_118.77965']                #type(MyTrafficFrame.values) == <type 'numpy.ndarray'>  



def spatial_corr(MyTrafficFrame):
    spatial_corr = MyTrafficFrame.corr()                                      #spatial correlation 
    return spatial_corr


def autocorrelation(x,lags):                                                  #Temporal correlation  
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
            /(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    return result

def normalization(x):                                                         #type of input is <type 'numpy.ndarray'>
    min_max_scaler = preprocessing.MinMaxScaler()   
    return min_max_scaler.fit_transform(x)


def normalization_2(x):
    for i in range(np.shape(x)[1]):
        x[:,i] =  0.5*(np.tanh(0.01*(x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])) + 1)
        #x[i] = x[i]/np.max(x)
    return x

def normalization_1(x):
    x =  0.5*(np.tanh(0.01*(x - np.mean(x))/np.std(x)) + 1)
    return x


#def normalization(trafficFrame):
#    zscore = lambda x: 0.5*(np.tanh(0.01(x - np.mean(x))/np.std(x)) + 1)
#    norm_trafficFrame = trafficFrame.apply(zscore)
#    return norm_trafficFrame





def corrupt(x):
    """Take an input tensor and add uniform masking.
    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),\
                    minval=0,maxval=2,dtype=tf.int32), tf.float32))





##Autoencoder definition
def autoencoder(dimensions=[784, 512, 256, 64]):
    """Build a deep autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% input to the network
    corrupt_prob = tf.placeholder(tf.float32,[1])
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = corrupt(x) * corrupt_prob + x*(1-corrupt_prob)

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.zeros([n_input, n_output]))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output

    # %% latent representation
    z = current_input   ##compressed representation
    encoder.reverse()

    # %% Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output

    # %% now have the reconstruction through the network
    y = current_input

    # %% cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    #cost = tf.nn.sigmoid_cross_entropy_with_logits(y,x)
    #cost = tf.reduce_mean(-tf.reduce_sum(x * tf.log(y),reduction_indices=[1]))

    return {'x': x, 'z': z, 'y': y, 'corrupt_prob':corrupt_prob,'cost': cost}





def get_compressed_representation():
    trafficFrame,target_trafficFrame = loadDataSet()
    traffic_mat = trafficFrame.values                                             #shape(traffic_mat) = (2861,19)
    target_traffic = target_trafficFrame.values                                   #shape(target_traffic) = (2861,)

    traffic_mat = normalization(traffic_mat)
    target_traffic =  normalization(target_traffic.reshape(-1,1))

    ae = autoencoder(dimensions=[19, 2])                                           ##dimension[0] = number of cells
    learning_rate = 0.008
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_size = 128
    n_epochs = 40

    for epoch_i in range(n_epochs):
        for batch_i in range(22):
            trainSet = traffic_mat[batch_i*batch_size:(batch_i+1)*batch_size,:]
            sess.run(optimizer,feed_dict={ae['x']:trainSet,ae['corrupt_prob']: [1.0]})
    mat_compressed =  sess.run(ae['z'],feed_dict={ae['x']:traffic_mat,ae['corrupt_prob']: [1.0]}) ##shape(mat_compressed) = (2861,2)

    ##generate mat for LSTM inputs
    mat_for_predict = np.column_stack((target_traffic,mat_compressed))                            ## shape(mat_for_predict) = (2861,3)
    sess.close()
    return mat_for_predict,target_traffic                                                        ##  shape(target_traffic) = (2861,)


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.scalar_summary('cost', self.cost)

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.sub(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)




def get_trainSet():
    global TIME_STEPS,ONEDAY_LENGTH
    mat_for_predict,target_traffic = get_compressed_representation()
    traffic_length = len(target_traffic)
    xs = np.arange(0, traffic_length)
    seq = np.zeros((traffic_length - ONEDAY_LENGTH, TIME_STEPS,3))                       
    res = np.zeros((traffic_length - ONEDAY_LENGTH, TIME_STEPS,1))                        
    for i in range(0,traffic_length - TIME_STEPS*2):
        seq[i] = mat_for_predict[xs[i:i+TIME_STEPS]].reshape(TIME_STEPS,3)
        res[i] = target_traffic[xs[i+TIME_STEPS:i+2*(TIME_STEPS)]].reshape(TIME_STEPS,1)
    return seq,res,xs                                                           ##shape(seq) = (2861-96, 96, 3)
                                                                                ##shape(res) = (2861-96, 96, 1)
                                                                                ##shape(xs) = (2861-96,)


def get_batch():
    global BATCH_SIZE, TIME_STEPS, BATCH_START
    seq,res,xs = get_trainSet()
    sub_seq = seq[BATCH_START:BATCH_START+BATCH_SIZE]
    sub_res = res[BATCH_START:BATCH_START+BATCH_SIZE]
    BATCH_START += BATCH_SIZE
    return sub_seq,sub_res,xs                                                  
                                       



if __name__ =='__main__':
    BATCH_START = 0
    TIME_STEPS = 96
    BATCH_SIZE = 100
    INPUT_SIZE = 3
    OUTPUT_SIZE = 1
    CELL_SIZE = 10
    ONEDAY_LENGTH = 96
    LR = 0.01
    n_epochs = 20
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs_15min", sess.graph)
    sess.run(tf.initialize_all_variables())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

##training the model
    for epoch_j in range(n_epochs):
        BATCH_START = 0
        for i in range(20):
            sub_seq, sub_res, xs = get_batch()
            if i == 0:
                feed_dict = {
                        model.xs: sub_seq,
                        model.ys: sub_res,
                        # create initial state
                }
            else:
                feed_dict = {
                        model.xs: sub_seq,
                        model.ys: sub_res,
                        model.cell_init_state: state    # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],feed_dict=feed_dict)



##test model2
    seq,res,xs = get_trainSet()
    feed_dict = {
                model.xs:seq[2000:2700:7].reshape(-1,TIME_STEPS,3)
                }
    
    res = res[2000:2700:7].reshape(-1,TIME_STEPS,1)                                          ##the ground truth result to compare
    res = res[::12]
    pred = sess.run(model.pred,feed_dict=feed_dict).reshape(-1,TIME_STEPS,1)
    pred = pred[::12]
    
    plt.plot(np.arange(len(res[:].flatten())), res[:].flatten(), 'r', np.arange(len(res[:].flatten())), pred.flatten()[:], 'b--')
    plt.ylim((0,1.2))
    plt.show()

    res = res[:].flatten()
    pred = pred.flatten()
    print metrics.mean_squared_error(res,pred)              ##MSE = 0.0128584471636
    print metrics.mean_absolute_error(res,pred)             ##MAE = 0.0795121250112
