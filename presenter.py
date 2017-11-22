from datetime import datetime
from matplotlib import pyplot as plt


def save_plot(x_labels, y_labels_1, y_labels_2, x_title, y_title, param_name):
    plt.plot(x_labels, y_labels_1)
    plt.plot(x_labels, y_labels_2)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    now = datetime.now()
    date_repr = f'{now.date()}_{now.time()}'.replace(':', '-').replace('.', '-')
    plt.savefig(f'./plots/{param_name}_{date_repr}.png')
    plt.clf()
