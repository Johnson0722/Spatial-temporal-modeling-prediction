#coding:utf-8

import numpy as np
import pandas as pd
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext,SQLContext
import tensorflow as tf


def preprocessing():
    ##---------load user information------------##
    Users_df = sqlContext.read.json("/home/johnson/User_behavior_analysis/Data_1hour/part-r-00000-58453bdb-ae0e-42c5-909b-271b2125aaef") ##type(user_info) = "pyspark.sql.dataframe.DataFrame"

    Service_info = Users_df.select("ServiceMat")
    Service_df = Service_info.toPandas()     ##[1778 rows x 1 columns]
    Service_array = np.zeros((1778,360))     ##shape = (1778,360)
    for i in range(1778):
        Service_array[i] = np.array(Service_df.ix[i][0]).flatten()      ##transform into array



    ##----------Kmeans clustering----------##
    n_clusters = 6
    Service_array_rdd = sc.parallelize(Service_array)
    model = KMeans.train(Service_array_rdd,n_clusters,maxIterations=10,initializationMode="random",\
                        seed=50,initializationSteps=5,epsilon=1e-4)

    UserServiceLabel =np.zeros(1778)                ##Each user's Service label
    i = 0
    for userSer in Service_array:
        UserServiceLabel[i] = model.predict(userSer)
        i = i + 1


    ##-----------Clustering Users---------##
    label_index = []                                ##Label_index中每个元素都是array类型
    for cluster_label in range(model.k):
        label_index.append(np.where(UserServiceLabel == cluster_label))

    UserCluster0 = Service_array[label_index[0]]     #np.shape = (328,360)
    UserCluster1 = Service_array[label_index[1]]     #np.shape = (276,360)
    UserCluster2 = Service_array[label_index[2]]     #np.shape = (128,360)
    UserCluster3 = Service_array[label_index[3]]     #np.shape = (101,360)
    UserCluster4 = Service_array[label_index[4]]     #np.shape = (139,360)
    UserCluster5 = Service_array[label_index[5]]     #np.shape = (803,360)

    num_Cluster0 = np.shape(UserCluster0)[0]
    num_Cluster1 = np.shape(UserCluster1)[0]
    num_Cluster2 = np.shape(UserCluster2)[0]
    num_Cluster3 = np.shape(UserCluster3)[0]
    num_Cluster4 = np.shape(UserCluster4)[0]
    num_Cluster5 = np.shape(UserCluster5)[0]

    '''
    UserCluster0 = UserCluster0.reshape(-1,24,15)
    UserCluster1 = UserCluster1.reshape(-1,24,15)
    UserCluster2 = UserCluster2.reshape(-1,24,15)
    UserCluster3 = UserCluster3.reshape(-1,24,15)
    UserCluster4 = UserCluster4.reshape(-1,24,15)
    UserCluster5 = UserCluster5.reshape(-1,24,15)
    '''
    return UserCluster0,num_Cluster0






##------------Service Modeling&&Prediction-------------##
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



def get_compressed_representation(UserCluster,num_Cluster):                             ##Analysis the Cluster0
    target_user = UserCluster[24]                               ##random select a user
    ae = autoencoder(dimensions=[num_Cluster, 50, 10])      ##dimension[0] = number of users in the Cluster
    learning_rate = 0.08                                        ##shape(UserCluster0) = (?, 360)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 120
    n_epochs = 40
    UserCluster = UserCluster.transpose()                     ##after transpose, shape(UserCluster0) = (360, ?)
    for epoch_i in range(n_epochs):
        for batch_i in range(3):
            trainSet = UserCluster[batch_i*batch_size:(batch_i+1)*batch_size,:]
            sess.run(optimizer,feed_dict={ae['x']:trainSet,ae['corrupt_prob']: [1.0]})
    UserCluster_features =  sess.run(ae['z'],feed_dict={ae['x']:UserCluster,ae['corrupt_prob']: [1.0]}) ##shape(mat_compressed) = (946,2)

    ##generate mat for LSTM inputs
    Usermat_for_predict = np.column_stack((target_user,UserCluster_features))                            ## shape(Usermat_for_predict) = (360,11)
    sess.close()
    return Usermat_for_predict,target_user                                                               ##  shape(target_user) = (360,)





'''
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size])
        self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size])

        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()

        self.compute_cost()
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)


    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size])  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = tf.Variable(tf.random_normal([self.input_size, self.cell_size], stddev = 0.1))
        # bs (cell_size, )
        bs_in = tf.Variable(tf.constant(0.0,shape = [self.cell_size,]))
        # l_in_y = (batch * n_steps, cell_size)
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size])


    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=0.1)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        Ws_out = tf.Variable(tf.random_normal([self.cell_size, self.output_size],stddev = 1.0))
        bs_out = tf.Variable(tf.constant(0.0, shape = [self.output_size, ]))
        # shape = (batch * steps, output_size)
        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out


    def compute_cost(self):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1])],
            [tf.reshape(self.ys, [-1])],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error)

        self.cost = tf.div(tf.reduce_sum(losses),self.batch_size)
            

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.sub(y_pre, y_target))




###---------------------1 hour history for 1 hour future-----------###
def get_trainSet():
    global TIME_STEPS
    UserCluster, num_Cluster = preprocessing()      
    Usermat_for_predict,target_user = get_compressed_representation(UserCluster,num_Cluster)       ## shape(mat_for_predict) = (360,11)
    Service_length = len(target_user)                                                              ## shape(target_user) = (360,)
    xs = np.arange(0, Service_length)
    seq = np.zeros((Service_length - TIME_STEPS, TIME_STEPS,11))                                  
    res = np.zeros((Service_length - TIME_STEPS, TIME_STEPS,1))                        
    for i in range(0,Service_length - 2*TIME_STEPS):
        seq[i] = Usermat_for_predict[xs[i:i+TIME_STEPS]].reshape(TIME_STEPS,11)
        res[i] = target_user[xs[i+TIME_STEPS:i+2*(TIME_STEPS)]].reshape(TIME_STEPS,1)
    return seq,res,xs                                                                              ##shape(seq) = (345,15,11)
                                                                                                   ##shape(res) = (345,15,1)


def get_batch():
    global BATCH_SIZE, TIME_STEPS, BATCH_START
    seq,res,xs = get_trainSet()
    sub_seq = seq[BATCH_START:BATCH_START+BATCH_SIZE]
    sub_res = res[BATCH_START:BATCH_START+BATCH_SIZE]
    BATCH_START += BATCH_SIZE
    return sub_seq,sub_res,xs 







if __name__  == '__main__':
    ##----------initialize pyspark--------------##
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    ##----------set parameters-----------##
    BATCH_START = 0
    TIME_STEPS = 15
    BATCH_SIZE = 165               ## 1 hour history predict 1 hour
    INPUT_SIZE = 11
    OUTPUT_SIZE = 1
    CELL_SIZE = 12
    LR = 0.01
    n_epochs = 20
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##--------training the model----------##
    for epoch_j in range(n_epochs):
        BATCH_START = 0
        for i in range(2):
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

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],feed_dict=feed_dict)


    ##---------#test model---------##                                               ##shape(seq) = (345, 15, 11)
    seq,res,xs = get_trainSet()                                                      ##shape(res) = (345, 15, 1)
    feed_dict = {                                                                     
                model.xs:seq[180:345:1].reshape(-1,TIME_STEPS,11)
                }
    
    res = res[180:345:1].reshape(-1,TIME_STEPS,1)                                            ##the ground truth result to compare
    res = res[::15]                                                                          ##shape(res) = (11,15,1)
    pred = sess.run(model.pred,feed_dict=feed_dict).reshape(-1,TIME_STEPS,1)
    pred = pred[::15]                                                                        ##shape(pred) = (11,15,1)

    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0

    res = res.reshape(11,15)
    pred = pred.reshape(11,15)

    res = pd.DataFrame(res,index = [14,15,16,17,18,19,20,21,22,23,24])
    pred = pd.DataFrame(pred,index = [14,15,16,17,18,19,20,21,22,23,24])


    print pred
    print res


    TP = np.sum(np.sum((pred == 1) & (res == 1)))
    print float(np.sum(np.sum(pred == res)))/(15*11)       ##total accuracy
    print float(TP)/(np.sum(np.sum(pred == 1)))            ##precision
    print float(TP)/(np.sum(np.sum(res == 1)))             ##recall rate
'''




##-----------2 hour history for 1 hour future-------------##

class LSTMRNN(object):
    def __init__(self, n_steps1,n_steps2, input_size, output_size, cell_size, batch_size):
        self.n_steps1 = n_steps1
        self.n_steps2 = n_steps2
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        self.xs = tf.placeholder(tf.float32, [None, n_steps1, input_size])
        self.ys = tf.placeholder(tf.float32, [None, n_steps2, output_size])


        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()

        self.compute_cost()
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)


    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size])  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = tf.Variable(tf.random_normal([self.input_size, self.cell_size], stddev = 0.1))
        # bs (cell_size, )
        bs_in = tf.Variable(tf.constant(0.0,shape = [self.cell_size,]))
        # l_in_y = (batch * n_steps1, cell_size)
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps1, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps2, self.cell_size])


    def add_cell(self):
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=0.1)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        Ws_out = tf.Variable(tf.random_normal([self.cell_size, self.output_size],stddev = 1.0))
        bs_out = tf.Variable(tf.constant(0.0, shape = [self.output_size, ]))
        # shape = (batch * steps, output_size)
        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out


    def compute_cost(self):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1])],
            [tf.reshape(self.ys, [-1])],
            [tf.ones([self.batch_size * self.n_steps2], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error)

        self.cost = tf.div(tf.reduce_sum(losses),self.batch_size)
            

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.sub(y_pre, y_target))



def get_trainSet():
    global TIME_STEPS
    UserCluster, num_Cluster = preprocessing()      
    Usermat_for_predict,target_user = get_compressed_representation(UserCluster,num_Cluster)       ## shape(mat_for_predict) = (360,11)
    Service_length = len(target_user)                                                              ## shape(target_user) = (360,)
    xs = np.arange(0, Service_length)
    seq = np.zeros((Service_length - TIME_STEPS*2, TIME_STEPS*2,11))                                  
    res = np.zeros((Service_length - TIME_STEPS*2, TIME_STEPS,1))                        
    for i in range(0,Service_length - 3*TIME_STEPS):
        seq[i] = Usermat_for_predict[xs[i:i+TIME_STEPS*2]].reshape(2*TIME_STEPS,11)
        res[i] = target_user[xs[i+TIME_STEPS*2:i+TIME_STEPS*3]].reshape(TIME_STEPS,1)
    return seq,res,xs                                                                              ##shape(seq) = (330,30,11)
                                                                                                   ##shape(res) = (330,15,1)
                                                                                                   

def get_batch():
    global BATCH_SIZE, TIME_STEPS, BATCH_START
    seq,res,xs = get_trainSet()
    sub_seq = seq[BATCH_START:BATCH_START+BATCH_SIZE]
    sub_res = res[BATCH_START:BATCH_START+BATCH_SIZE]
    BATCH_START += BATCH_SIZE
    return sub_seq,sub_res,xs 







if __name__  == '__main__':
    ##----------initialize pyspark--------------##
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    ##----------set parameters-----------##
    BATCH_START = 0
    TIME_STEPS1 = 30
    TIME_STEPS2 = 15
    TIME_STEPS = 15
    BATCH_SIZE = 105               ## 1 hour history predict 1 hour
    INPUT_SIZE = 11
    OUTPUT_SIZE = 1
    CELL_SIZE = 12
    LR = 0.01
    n_epochs = 20
    model = LSTMRNN(TIME_STEPS1,TIME_STEPS2, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##--------training the model----------##
    for epoch_j in range(n_epochs):
        BATCH_START = 0
        for i in range(3):
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


    ##---------test model---------##                                               ##shape(seq) = (330, 30, 11)
    seq,res,xs = get_trainSet()                                                      ##shape(res) = (330, 15, 1)
    feed_dict = {                                                                     
                model.xs:seq[225:330:1].reshape(-1,TIME_STEPS*2,11)
                }
    
    res = res[225:330:1].reshape(-1,TIME_STEPS,1)                                            ##the ground truth result to compare
    res = res[::15]                                                                          ##shape(res) = (7,15,1)
    pred = sess.run(model.pred,feed_dict=feed_dict).reshape(-1,TIME_STEPS,1)
    pred = pred[::15]                                                                        ##shape(pred) = (7,15,1)

    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0

    res = res.reshape(11,15)
    pred = pred.reshape(11,15)

    res = pd.DataFrame(res,index = [18,19,20,21,22,23,24])
    pred = pd.DataFrame(pred,index = [18,19,20,21,22,23,24])


    print pred
    print res


    TP = np.sum(np.sum((pred == 1) & (res == 1)))          ##True positive
    print float(np.sum(np.sum(pred == res)))/(15*7)        ##total accuracy
    print float(TP)/(np.sum(np.sum(pred == 1)))            ##precision
    print float(TP)/(np.sum(np.sum(res == 1)))             ##recall rate


'''
##----------------3 hour history for 1 hour future---------------##

def get_trainSet():
    global TIME_STEPS
    UserCluster, num_Cluster = preprocessing()      
    Usermat_for_predict,target_user = get_compressed_representation(UserCluster,num_Cluster)       ## shape(mat_for_predict) = (360,11)
    Service_length = len(target_user)                                                              ## shape(target_user) = (360,)
    xs = np.arange(0, Service_length)
    seq = np.zeros((Service_length - TIME_STEPS, TIME_STEPS,11))                                  
    res = np.zeros((Service_length - TIME_STEPS, TIME_STEPS,1))                        
    for i in range(0,Service_length - 2*TIME_STEPS):
        seq[i] = Usermat_for_predict[xs[i:i+TIME_STEPS]].reshape(TIME_STEPS,11)
        res[i] = target_user[xs[i+TIME_STEPS:i+2*(TIME_STEPS)]].reshape(TIME_STEPS,1)
    return seq,res,xs                                                                              ##shape(seq) = (345,15,11)
                                                                                                   ##shape(res) = (345,15,1)


def get_batch():
    global BATCH_SIZE, TIME_STEPS, BATCH_START
    seq,res,xs = get_trainSet()
    sub_seq = seq[BATCH_START:BATCH_START+BATCH_SIZE]
    sub_res = res[BATCH_START:BATCH_START+BATCH_SIZE]
    BATCH_START += BATCH_SIZE
    return sub_seq,sub_res,xs 







if __name__  == '__main__':
    ##----------initialize pyspark--------------##
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    ##----------set parameters-----------##
    BATCH_START = 0
    TIME_STEPS = 15
    BATCH_SIZE = 165               ## 1 hour history predict 1 hour
    INPUT_SIZE = 11
    OUTPUT_SIZE = 1
    CELL_SIZE = 12
    LR = 0.01
    n_epochs = 20
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##--------training the model----------##
    for epoch_j in range(n_epochs):
        BATCH_START = 0
        for i in range(2):
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


    ##---------#test model2---------##                                               ##shape(seq) = (345, 15, 11)
    seq,res,xs = get_trainSet()                                                      ##shape(res) = (345, 15, 1)
    feed_dict = {                                                                     
                model.xs:seq[180:345:1].reshape(-1,TIME_STEPS,11)
                }
    
    res = res[180:345:1].reshape(-1,TIME_STEPS,1)                                            ##the ground truth result to compare
    res = res[::15]                                                                          ##shape(res) = (11,15,1)
    pred = sess.run(model.pred,feed_dict=feed_dict).reshape(-1,TIME_STEPS,1)
    pred = pred[::15]                                                                        ##shape(pred) = (11,15,1)

    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0

    res = res.reshape(11,15)
    pred = pred.reshape(11,15)

    res = pd.DataFrame(res,index = [14,15,16,17,18,19,20,21,22,23,24])
    pred = pd.DataFrame(pred,index = [14,15,16,17,18,19,20,21,22,23,24])


    print pred
    print res


    TP = np.sum(np.sum((pred == 1) & (res == 1)))          ##True positive
    print float(np.sum(np.sum(pred == res)))/(15*11)       ##total accuracy
    print float(TP)/(np.sum(np.sum(pred == 1)))            ##precision
    print float(TP)/(np.sum(np.sum(res == 1)))             ##recall rate
'''

