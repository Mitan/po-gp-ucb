from src.enum.MethodEnum import MethodEnum


class MethodDescriptor:
    def __init__(self, method_type, epsilon_log=0, r=None):

        # log of DP parameter epsilon
        self.epsilon_log = epsilon_log

        # dimension of random projection
        self.r = r

        # string for printing in logging
        if method_type == MethodEnum.UCB:
            self.method_string = 'Non-private GP-UCB'
        elif method_type == MethodEnum.ODP_GP_UCB:
            assert epsilon_log is not None, "For transformed method epsilon must be set"
            assert r is not None, "For transformed method r must be set"
            self.method_string = 'ODP-GP-UCB with r = {} and log_epsilon = {}'.format(self.r, self.epsilon_log)
        else:
            raise ValueError("Unknown method")

        # define is method is running BO on original dataset or transformed dataset
        self.method_type = method_type

        if method_type == MethodEnum.UCB:
            self.plotting_method_string = "Non-private"
        else:
            self.plotting_method_string = r"$r = {}, \ \epsilon = \exp({})$".format(self.r, self.epsilon_log)
        # elif r_or_epsilon == 'r':
        #     self.plotting_method_string =  r"$r = {}$".format(self.r)
        # elif r_or_epsilon == 'epsilon':
            # self.plotting_method_string = r"$r = {}, \ \epsilon = \exp({})$".format(self.r, self.epsilon_log)
        # else:
        #     raise ValueError("Unknown r_or_epsilon")
