import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CreateMatplotXdata:
    def __init__(self):
        self.years = mdates.YearLocator()  # every year
        self.months = mdates.MonthLocator()  # every month
        self.dates = mdates.DateLocator()
        self.date_fmt = mdates.DateFormatter('%Y-%m-%d')

    def create_matplot_x_data(self):
        fig, ax = plt.subplots()
        # format the ticks

        ax.xaxis.set_major_locator(self.years)
        ax.xaxis.set_major_formatter(self.date_fmt)
        ax.xaxis.set_minor_locator(self.months)
        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x
        ax.grid(True)
        return ax
