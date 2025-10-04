import sys
import json


def called_function(dict):
    # for key, value in dict.items():
    #     print(key, value)
    print(dict)


if __name__ == "__main__":
    path_to_json_file_with_dict = sys.argv[1]

    with open(path_to_json_file_with_dict, "r") as f:
        dict = json.load(f)
    called_function(dict)
