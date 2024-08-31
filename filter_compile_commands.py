# this script trims compile_commands.json to contain no files from triton/ folder
# to avoid analyzing code from the submodule
import json
import os

input_file = 'triton/python/compile_commands.json'
output_file = 'compile_commands.json'

def filter_compile_commands(input_file, output_file):
    with open(input_file, 'r') as f:
        compile_commands = json.load(f)

    filtered_commands = [
        entry for entry in compile_commands
        if 'triton_shared/triton' not in entry['file']
    ]

    with open(output_file, 'w') as f:
        json.dump(filtered_commands, f, indent=2)

    print(f"Filtered compile_commands.json written to {output_file} with {len(filtered_commands)} entries.")

if __name__ == "__main__":
    filter_compile_commands(input_file, output_file)
