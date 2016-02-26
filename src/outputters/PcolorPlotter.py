import matplotlib.pyplot as plt

class PcolorPlotter:
    def __init__(self, config):
        self.interval = config['plotting_interval']
        self.plot_to_screen = config['plot_to_screen']
        self.plot_to_file = config['plot_to_file']

    def output(self, timestep, x, y, field):
        if (timestep % self.interval) == 0 and self.plot_to_screen is True:
            plt.clf()
            plt.pcolor(x, y, field)
            plt.colorbar()
            plt.draw()
            plt.show(block=False)
