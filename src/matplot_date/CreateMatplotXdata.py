import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CreateMatplotXdata:
    def __init__(self, xdata):
        self.xdata = xdata
        self.years = mdates.YearLocator()  # every year
        self.months = mdates.MonthLocator()  # every month
        self.dates = mdates.DateLocator()
        self.date_fmt = mdates.DateFormatter('%Y-%m-%d')

    def create_matplot_x_data(self):

        fig, ax = plt.subplots()
        # format the ticks
        datemin = np.datetime64(self.xdata.iloc[0], 'D')
        datemax = np.datetime64(self.xdata.iloc[-1], 'D') + np.timedelta64(1, 'D')
        ax.set_xlim(datemin, datemax)

        matplotlib.rc('xtick', labelsize=15)
        matplotlib.rc('ytick', labelsize=15)
        matplotlib.rc('legend', fontsize=15)
        matplotlib.rc('axes', titlesize=15)
        matplotlib.rc('axes', labelsize=15)

        ax.xaxis.set_major_locator(self.years)
        ax.xaxis.set_major_formatter(self.date_fmt)
        ax.xaxis.set_minor_locator(self.months)
        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x
        ax.grid(True)

        return ax
