

class LearningParam:
    """
        Controls a Learning Parameter including exponential rate of decay etc.
    """

    def __init__(self, init_value, end_value_ratio, n):
        """
            Constructor
        :param init_value: initial value
        :param end_value_ratio: ratio between the end value and init_value
        :param n: expected number of calls to the parameter
        """
        self.decay = 1.0*end_value_ratio**(1.0/n)
        self.init_value = init_value
        self.value = init_value

    def get_value(self):
        self.value *= self.decay
        return self.value

    def get_value_no_decay(self):
        return self.value

