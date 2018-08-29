import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import data_pre
import visualization

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_data(batch_size=60, time_step=20, train_begin=0, train_end=1500):
    #标准化
    normalization = MinMaxScaler(feature_range=(0,1))
    normalized_data = normalization.fit_transform(data[:,1].reshape(-1,1))
    normalized_train_data = normalized_data[train_begin:train_end].tolist()
    normalized_test_data = normalized_data[train_end:].tolist()
    
    #训练集，训练数据后移一个时间单位作为label
    batch_index = []
    train_x, train_y = [], []
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step]
        y = normalized_train_data[i+1:i+time_step+1]
        train_x.append(x)
        train_y.append(y)
    batch_index.append((len(normalized_train_data)-time_step))
    
    #测试集
    test_size = (len(normalized_test_data)+time_step-1)//time_step   #按time_step分割
    test_x, test_y = [], []
    for i in range(test_size):
        x = normalized_test_data[i*time_step:(i+1)*time_step]
        y = normalized_test_data[i*time_step+1:(i+1)*time_step+1]
        test_x.append(x)
        test_y.append(y)
    test_y[-1].append([test_y[-1][-1][-1]])
    
    return batch_index, train_x, train_y, test_x, test_y, normalization
    
def lstm_model(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    
    input = tf.reshape(X, [-1,input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])   #将tensor转成3维，作为lstm cell的输入
    basic_cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit, state_is_tuple=True)
    multi_cell = tf.contrib.rnn.MultiRNNCell([basic_cell]*layer_num, state_is_tuple=True)
    init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(multi_cell, input_rnn, initial_state=init_state, dtype=tf.float32)   #final_states是最后一个cell的结果
    output = tf.reshape(output_rnn,[-1,rnn_unit])
    
    w_out = weights['out']
    b_out = biases['out']
    model = tf.matmul(output,w_out) + b_out
    return model, final_states
    
def train_lstm(steps = 500, batch_size=60, time_step=20, train_begin=0, train_end=1500):
    X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y = tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index, train_x, train_y, test_x, test_y, normalization = get_data(batch_size, time_step, train_begin, train_end)
    model, final_states = lstm_model(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(model,[-1])-tf.reshape(Y,[-1])))  #损失函数
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)  #优化
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #训练500轮
    for step in range(steps):
        for batch in range(len(batch_index)-1):
            final_states, _loss = sess.run([train_op,loss], feed_dict={X:train_x[batch_index[batch]:batch_index[batch+1]],Y:train_y[batch_index[batch]:batch_index[batch+1]]})
        if (step+1) % 100 == 0:
            print('step:', step+1, '   loss:', _loss)
    #预测
    test_predict=[]
    for step in range(len(test_x)):
        predict = sess.run(model, feed_dict={X:[test_x[step]]}).reshape((-1))
        test_predict.extend(predict)
    #逆标准化
    test_predict = normalization.inverse_transform(np.array(test_predict).reshape(-1,1))
    test_y = normalization.inverse_transform(np.array(test_y).reshape(-1,1))
    #平均误差与标准差
    mae = mean_absolute_error(test_predict, test_y)
    rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    print ('mae:', mae, '   rmse:', rmse)
    
    return test_predict
    
if __name__ == '__main__':
    #数据预处理
    data = data_pre.read_data('data/time_temperture_data.csv')
    data = data_pre.data_interpolation(data)
    data = data_pre.reduce_data(data)
    
    #超参数
    rnn_unit = 10
    layer_num = 8
    input_size = 1
    output_size = 1
    learning_rate = 0.0006
    
    tf.reset_default_graph()
    #输入层、输出层权重、偏置
    weights={'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
             'out':tf.Variable(tf.random_normal([rnn_unit,1]))}
    biases={'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[1,]))}
    #训练+预测
    test_predict = train_lstm(steps = 500, batch_size=60, time_step=20, train_begin=0, train_end=1500)
    #可视化
    visualization.show_result(observed_times=range(0,150000,100), observed_data=data[0:1500,1],
                              predicted_times=range(150000,180000,100), predicted_data=test_predict.reshape(300))
    visualization.save_result('predict_result2.jpg',
                              observed_times=range(0,150000,100), observed_data=data[0:1500,1],
                              predicted_times=range(150000,180000,100), predicted_data=test_predict.reshape(300))
