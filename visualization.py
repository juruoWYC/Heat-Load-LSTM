import matplotlib
import matplotlib.pyplot as plt

def show_result(observed_times=None, observed_data=None, evaluated_times=None, evaluated_data=None, predicted_times=None, predicted_data=None):
    plt.figure(figsize=(15, 5))
    cnfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    handles = list()
    if observed_times is not None:
        lines = plt.plot(observed_times, observed_data, label="数据", color="k")
        handles.append(lines[0])
    if evaluated_times is not None:
        lines = plt.plot(evaluated_times, evaluated_data, label="拟合", color="g")
        handles.append(lines[0])
    if predicted_times is not None:
        lines = plt.plot(predicted_times, predicted_data, label="预测", color="r")
        handles.append(lines[0])
    
    plt.legend(handles=handles, loc="upper left", prop=cnfont)
    plt.show()
    
def save_result(file_name, observed_times=None, observed_data=None, evaluated_times=None, evaluated_data=None, predicted_times=None, predicted_data=None):
    plt.figure(figsize=(15, 5))
    cnfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    handles = list()
    if observed_times is not None:
        lines = plt.plot(observed_times, observed_data, label="数据", color="k")
        handles.append(lines[0])
    if evaluated_times is not None:
        lines = plt.plot(evaluated_times, evaluated_data, label="拟合", color="g")
        handles.append(lines[0])
    if predicted_times is not None:
        lines = plt.plot(predicted_times, predicted_data, label="预测", color="r")
        handles.append(lines[0])
    
    plt.legend(handles=handles, loc="upper left", prop=cnfont)
    plt.savefig(file_name)
