import numpy as np

class Replicate():

    def __init__(self, nsel = None, n = None, selected_points = None):

        self.nsel = nsel
        if selected_points is not None:
            self.selected_points = selected_points
        else:
            self.selected_points = np.random.choice(list(range(n)), nsel, replace=False)