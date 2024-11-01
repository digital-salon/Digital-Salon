import subprocess


def process_string(input_str):
    command = [
        "conda run -n control-v11 python ../extern/controlnet/canny_controlnet_api.py"
        + " "
        + input_str
    ]
    print(command)

    # Run the command
    result = subprocess.run(command, shell=True)

    return f"Processed: {input_str}"
