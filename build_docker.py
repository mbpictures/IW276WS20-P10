import os
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use this script to build the docker image")
    parser.add_argument("name", help="The repository name of the docker image", type=str, nargs='?', default="iw276ws20-p10")
    parser.add_argument("tag", help="The tag of the docker image", type=str, nargs='?', default="0.1")
    args = parser.parse_args()

    print(f"DEBUG: BUILDING IMAGE WITH NAME: {args.name}")
    print(f"DEBUG: BUILDING IMAGE WITH TAG: {args.tag}")

    subprocess.run([
        "sudo",
        "docker",
        "build",
        ".",
        "-t",
        f"{args.name}:{args.tag}"
    ])
