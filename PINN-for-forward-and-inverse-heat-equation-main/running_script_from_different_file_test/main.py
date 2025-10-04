import json
import os
import subprocess
import sys


def main_function():
    dir_of_this_file = os.path.abspath(os.path.dirname(__file__))

    test_dict = {
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "q_nn_kernel_initializer":                  "glorot_uniform",
        "loss_fn":                                  "L2_norm",
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (100, 100, 100),
        "write_loss_values_every_x_epochs":         10,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
    }

    path_to_json_file_with_dict = os.path.join(".", "params.json")
    with open(path_to_json_file_with_dict, "w") as f:
        json.dump(test_dict, f)

    path_to_script_to_be_called = os.path.abspath(os.path.join(dir_of_this_file, "called file.py"))

    subprocess.run([sys.executable, path_to_script_to_be_called, path_to_json_file_with_dict])

    # os.remove(path_to_json_file_with_dict)
