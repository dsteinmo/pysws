from pprint import pprint


class ConsoleLogger:
    def __init__(self, config):
        self.interval = config['logging_interval']

    def output(self, timestep, message):
        if (timestep % self.interval) == 0:
            pprint(message)
