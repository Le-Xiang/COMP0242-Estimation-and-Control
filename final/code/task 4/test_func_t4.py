# This file is used as a dataset fot all the experiments in Task 3-4
# The dataset specifies the initial states for the robot

import numpy as np

# Initial states for the robot
class Dataset():

    def __init__(self):
        self.x0_list = None
        self.eperiment_type = None
        self.num_experiments = 5

    def set_experiment_type(self, experiment_type):
        self.experiment_type = experiment_type
    
    def initial_states_by_experiment_type(self):
        if self.experiment_type == "position":
            self.x0_list = [
                np.array([2.0, 3.0, np.pi / 4]),
                np.array([3.0, 2.0, np.pi / 4]),
                np.array([-2.0, 3.0, np.pi / 4]),
                np.array([-3.0, 2.0, np.pi / 4]),
                np.array([2.0, -3.0, np.pi / 4]),
                np.array([3.0, -2.0, np.pi / 4]),
                np.array([-2.0, -3.0, np.pi / 4]),
                np.array([-3.0, -2.0, np.pi / 4]),
            ]
        elif self.experiment_type == "angle":
            self.x0_list = [
                np.array([2.0, 3.0, np.pi / 4]),
                np.array([2.0, 3.0, np.pi /2]),
                np.array([2.0, 3.0, 3*np.pi / 4]),
                np.array([2.0, 3.0, np.pi]),
                np.array([2.0, 3.0, -np.pi / 4]),
                np.array([2.0, 3.0, -np.pi / 2]),
                np.array([2.0, 3.0, -3*np.pi / 4]),
                np.array([2.0, 3.0, 0.0]),
            ]

        elif self.experiment_type == "distance":
            self.x0_list = [
                np.array([2.0, 3.0, np.pi / 4]),
                np.array([4.0, 6.0, np.pi / 4]),
                np.array([6.0, 9.0, np.pi / 4]),
                np.array([8.0, 12.0, np.pi / 4]),
            ]

        elif self.experiment_type == "sigma0":
            pass
        
        elif self.experiment_type == None:
            raise ValueError("Experiment type is not specified")
        else:
            raise ValueError("Invalid experiment type")
        
        return self.x0_list
        