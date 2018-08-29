import os
import numpy as np
from scipy.interpolate import interp1d

def read_data(csv_name):
    csv_data = np.loadtxt(csv_name, dtype=np.float64, delimiter=',')
    
    #时间序列整数化
    data_list = [[np.around(t[0], 0), t[1]] for t in csv_data[:]]
    new_data_list = list()
    t = data_list[0][1]
    cnt = 1
    for i in range(1, len(data_list)):
        if (data_list[i][0] == data_list[i-1][0]):
            t += data_list[i][1]
            cnt += 1
        else:
            new_data_list.append([int(data_list[i-1][0]), t/cnt])
            t = data_list[i][1]
            cnt = 1
    new_data_list.append([int(data_list[i][0]), t/cnt])
    
    return np.array(new_data_list)
    
def data_interpolation(data):
    max_value = np.int32(data[data.shape[0]-1][0])
    new_data = np.zeros(shape=(max_value+1, 2), dtype=np.float64)
    new_x = np.linspace(0, max_value, num=max_value+1, endpoint=True)
    #三次样条插值
    f = interp1d(data[:, 0], data[:, 1], kind = 'cubic')
    new_data[:, 0] = new_x
    new_data[:, 1] = f(new_x)
    
    return new_data
    
def reduce_data(data):
    #均匀取点，缩小数据规模至1/100
    data_x = list()
    data_y = list()
    cnt = 0
    for i in data:
        cnt += 1
        if (cnt % 100 == 0):
            cnt = 0
            data_x.append(i[0])
            data_y.append(i[1])
    
    new_data = np.zeros(shape=(len(data_x), 2), dtype=np.float64)
    new_data[:, 0] = data_x
    new_data[:, 1] = data_y
    return new_data
