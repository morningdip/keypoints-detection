import argparse
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors as colors

# Import the event accumulator from Tensorboard.
# Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False
# TF version < 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.python.summary import event_accumulator as ea
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version = 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.tensorboard.backend.event_processing import event_accumulator as ea
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version >= 1.3.0
if (not eventAccumulatorImported):
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version = Unknown
if (not eventAccumulatorImported):
    raise ImportError('Could not locate and import Tensorflow event accumulator.')

sns.set(style="darkgrid")
sns.set_context("paper")


def tb2plot(params):
    log_path = params['logdir']
    tag = params['tag']
    smooth_space = params['smooth']
    color_code = params['color']

    event_acc = ea.EventAccumulator(log_path)
    event_acc.Reload()

    # only support scalar now
    if tag == 'all':
        scalar_list = event_acc.Tags()['scalars']
    else:
        scalar_list = tag.split(',')

    x_list = []
    y_list = []
    x_list_raw = []
    y_list_raw = []

    for t in scalar_list:
        x = [int(s.step) for s in event_acc.Scalars(t)]
        y = [s.value for s in event_acc.Scalars(t)]

        # smooth curve
        x_ = []
        y_ = []

        for i in range(0, len(x), smooth_space):
            x_.append(x[i])
            y_.append(sum(y[i:i + smooth_space]) / float(smooth_space))

        x_.append(x[-1])
        y_.append(y[-1])
        x_list.append(x_)
        y_list.append(y_)

        # raw curve
        x_list_raw.append(x)
        y_list_raw.append(y)

    for i in range(len(x_list)):
        plt.figure(i)
        plt.subplot(111)
        plt.title(scalar_list[i])
        plt.plot(x_list_raw[i], y_list_raw[i], color=colors.to_rgba(color_code, alpha=0.4))
        plt.plot(x_list[i], y_list[i], color=color_code, linewidth=1.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', default='./logdir', type=str, help='logdir to event file')
    parser.add_argument('-t', '--tag', default='all', type=str, help='seperate tags with commas')
    parser.add_argument('-s', '--smooth', default=100, type=int, help='step size for average smoothing')
    parser.add_argument('-c', '--color', default='#4169E1', type=str, help='HTML code for the figure')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    tb2plot(params)
