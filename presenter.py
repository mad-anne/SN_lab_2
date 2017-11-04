from datetime import datetime
from matplotlib import pyplot as plt


def show_accuracies_plot(accuracies, x_labels):
    plt.plot(x_labels, accuracies)
    plt.xticks(x_labels)
    plt.xlabel('współczynnik uczenia')
    plt.ylabel('dokładność')
    plt.show()


def save_plot(x_labels, y_labels, x_title, y_title, param_name):
    plt.plot(x_labels, y_labels)
    plt.xticks(x_labels)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    now = datetime.now()
    date_repr = f'{now.date()}_{now.time()}'.replace(':', '-').replace('.', '-')
    plt.savefig(f'./plots/{param_name}_{date_repr}.png')
