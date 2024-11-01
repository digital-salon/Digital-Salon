import subprocess


def process_string(input_str):
    command = ["python ../extern/copilot/text_to_hair.py " + '"' + input_str + '"']
    print(command)
    # Run the command
    result = subprocess.run(command, shell=True)

    return f"Processed: {input_str}"
