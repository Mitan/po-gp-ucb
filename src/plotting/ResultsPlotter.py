import matplotlib

# Force matplotlib to not use any Xwindows backend.
from src.enum.DatasetEnum import DatasetEnum

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib import rc


# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=True)


class ResultGraphPlotter:

    def __init__(self, dataset_type, num_iterations):
        plt.rcParams["figure.figsize"] = [9, 6]

        self.dataset_type = dataset_type
        self.num_iterations = num_iterations

        self.axes = plt.axes(label="my_axes")

        # size of font at x and y label
        self.labels_font_size = 24

        # self.color_sequence = ['#e41a1c', '#4daf4a', '#984ea3',
        #                         '#a65628', '#f781bf',  '#377eb8','blue', '#ff7f00', 'black']

        self.color_sequence = [ '#f781bf','#a65628', '#984ea3',
                                 '#4daf4a', '#ff7f00','#377eb8', 'black', '#e41a1c' ]

        self.COLOR_LENGTH = len(self.color_sequence)
        self.markers = [  "2", "x", "|","*", "s",  "v", "^", "o"]

    def plot_results(self, results, plot_bars, output_file_name):
        if not results:
            return

        # 0 is width, 1 is height
        # results = [results[0]] + list(reversed(results[1:]))

        # for legends
        handles = []
        l = len(results)
        for i, result in enumerate(results):
            # hack to distingiuish private and non-private
            # color_index = 0 if i == 0 else i +  self.COLOR_LENGTH - l
            color_index =  i +  self.COLOR_LENGTH - l

            handle = self.__plot_one_method(color_index, result, plot_bars)
            handles.append(handle)

        plt.legend(handles=handles, loc=0, prop={'size': 12})

        max_y_value = max([max(result[1]) for result in results])
        min_y_value = min([min(result[1]) for result in results])
        self._plot_ticks_and_margins(min_y_value=min_y_value, max_y_value=max_y_value)
        plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')
        plt.clf()
        plt.close()

    def _plot_ticks_and_margins(self, min_y_value, max_y_value):

        self.axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        x_range = range(0, self.num_iterations + 1, 5)

        plt.xticks(x_range)
        plt.xlabel("No. of iterations", fontsize=self.labels_font_size)
        x_margin = 0.01 * (x_range[-1] - x_range[0])
        self.axes.set_xlim([x_range[0] - x_margin, x_range[-1] + x_margin])

        ylabel_name = "Simple regret"

        plt.ylabel(ylabel_name, fontsize=self.labels_font_size)

        ticks_interval = self._get_ticks_interval()

        y_ticks_min = round(min_y_value / ticks_interval)
        y_ticks_max = round(max_y_value / ticks_interval + 1)

        y_ticks  = ticks_interval * np.arange(y_ticks_min, y_ticks_max)
        plt.yticks(y_ticks)

        y_margin = 0.05 * (max_y_value - min_y_value)
        self.axes.set_ylim([min_y_value - y_margin, max_y_value + y_margin])

        tick_size = 15
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)

        # margins on x and y side
        self.axes.margins(x=0.035)
        self.axes.margins(y=0.035)

    # plot a method and return a legend handle
    def __plot_one_method(self, i, result, plot_bars):
        name = result[0]

        rewards = result[1]
        error_bars = result[2]

        time_steps = range(self.num_iterations + 1)

        marker_size = 5

        # line_style = '-' if i < 8 else '--'
        line_style = '-'
        marker_index = i if i < self.COLOR_LENGTH + 1  else 0
        color_index = i if (i < self.COLOR_LENGTH + 1) else 0


        # dirty hack to make markers unfilled by markerfacecolor="None"
        plt.plot(time_steps, rewards, lw=1.0, linestyle=line_style, marker=self.markers[marker_index],
                 markersize=marker_size,
                 markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=self.color_sequence[color_index],
                 color=self.color_sequence[color_index])

        # plt.plot(time_steps, rewards, lw=1.0, linestyle=line_style, color=self.color_sequence[i])

        if plot_bars:
            plt.errorbar(time_steps, rewards, yerr=error_bars, color=self.color_sequence[color_index], lw=0.1)

        patch = mlines.Line2D([], [], linestyle=line_style, color=self.color_sequence[color_index],
                              marker=self.markers[marker_index],
                              markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=self.color_sequence[color_index],
                              markersize=10, label=name)

        return patch

    def _get_ticks_interval(self):
        if self.dataset_type == DatasetEnum.Simulated:
            return 0.4
        elif self.dataset_type == DatasetEnum.HousePrice:
            # return 100.0
            return 0.4
        elif self.dataset_type == DatasetEnum.Loan:
            # return 100.0
            return 0.4
        elif self.dataset_type == DatasetEnum.Branin:
            # return 100.0
            return 0.4
        else:
            raise ValueError("Unknown dataset")
