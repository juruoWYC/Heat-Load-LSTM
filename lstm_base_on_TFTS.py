import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import data_pre
import visualization

import numpy as np
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import model as tft_model
from tensorflow.contrib.timeseries.python.timeseries import estimators as tft_estimators
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader


class _LSTMModel(tft_model.SequentialTimeSeriesModel):
    def __init__(self, num_units, num_features, dtype=tf.float32):
        super(_LSTMModel, self).__init__(
            train_output_names=['mean'],
            predict_output_names=['mean'],
            num_features=num_features,
            dtype=dtype)
        self._num_units = num_units
        self._lstm_cell = None
        self._lstm_cell_run = None
        self._predict_from_lstm_output = None
    
    def initialize_graph(self, input_statistics):
        super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
        self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
        self._lstm_cell_run = tf.make_template(
            name_='lstm_cell',
            func_=self._lstm_cell,
            create_scope_now_=True)
        
        self._predict_from_lstm_output = tf.make_template(
            name_='predict_from_lstm_output',
            func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
            create_scope_now_=True)
    
    def get_start_state(self):
        return (
            tf.zeros([], dtype=tf.int64),
            tf.zeros([self.num_features], dtype=self.dtype),
            [tf.squeeze(state_element, axis=0)
             for state_element
             in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])
    
    def _transform(self, data):
        mean, variance = self._input_statistics.overall_feature_moments
        return (data - mean) / variance
    
    def _de_transform(self, data):
        mean, variance = self._input_statistics.overall_feature_moments
        return data * variance + mean
    
    def _filtering_step(self, current_times, current_values, state, predictions):
        state_from_time, prediction, lstm_state = state
        with tf.control_dependencies(
                [tf.assert_equal(current_times, state_from_time)]):
          transformed_values = self._transform(current_values)
          predictions['loss'] = tf.reduce_mean(
              (prediction - transformed_values) ** 2, axis=-1)
          new_state_tuple = (current_times, transformed_values, lstm_state)
        return (new_state_tuple, predictions)
    
    def _prediction_step(self, current_times, state):
        _, previous_observation_or_prediction, lstm_state = state
        lstm_output, new_lstm_state = self._lstm_cell_run(
            inputs=previous_observation_or_prediction, state=lstm_state)
        next_prediction = self._predict_from_lstm_output(lstm_output)
        new_state_tuple = (current_times, next_prediction, new_lstm_state)
        return new_state_tuple, {'mean': self._de_transform(next_prediction)}
    
    def _imputation_step(self, current_times, state):
        return state


if __name__ == '__main__':
    #数据预处理
    data = data_pre.read_data('data/time_temperture_data.csv')
    data = data_pre.data_interpolation(data)
    data = data_pre.reduce_data(data)
    
    tf.logging.set_verbosity(tf.logging.INFO)

    data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: data[:, 0],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: data[:, 1]}
    reader = NumpyReader(data)
    
    #超参数
    rnn_unit = 64
    output_size = 1
    learning_rate = 0.0006
    
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=60, window_size=20)
    estimator = tft_estimators.TimeSeriesRegressor(model=_LSTMModel(num_features=output_size, num_units=rnn_unit),
                                                   optimizer=tf.train.AdamOptimizer(learning_rate))
    estimator.train(input_fn=train_input_fn, steps=2000)
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
    
    #预测
    (predictions,) = tuple(estimator.predict(input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation, steps=500)))

    #可视化
    visualization.show_result(observed_times=range(0,180000,100), observed_data=evaluation['observed'][0, :, :],
                              evaluated_times=range(0,180000,100), evaluated_data=evaluation['mean'][0],
                              predicted_times=range(180000,230000,100), predicted_data=predictions['mean'])
    visualization.save_result('predict_result.jpg',
                              observed_times=range(0,180000,100), observed_data=evaluation['observed'][0, :, :],
                              evaluated_times=range(0,180000,100), evaluated_data=evaluation['mean'][0],
                              predicted_times=range(180000,230000,100), predicted_data=predictions['mean'])
