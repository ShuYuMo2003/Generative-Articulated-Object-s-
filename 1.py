import os
current_file_path = os.path.realpath(__file__)
current_file_path = os.path.dirname(current_file_path)
print(current_file_path)