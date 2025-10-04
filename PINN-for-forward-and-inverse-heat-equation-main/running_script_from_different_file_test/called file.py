import sys
import json
import os


def called_function(dict):
    # for key, value in dict.items():
    #     print(key, value)
    print(dict)


if __name__ == "__main__":
    path_to_json_file_with_dict = sys.argv[1]
    print(f"File path from called script: {path_to_json_file_with_dict}")
    print(f"Absolute file path from called script: {os.path.abspath(path_to_json_file_with_dict)}")

    # This was here to test if memory gets cleared after the script ends - it does =D.
    # a = [32] * 400000000
    # input()

    with open(path_to_json_file_with_dict, "r") as f:
        dict = json.load(f)
    called_function(dict)
