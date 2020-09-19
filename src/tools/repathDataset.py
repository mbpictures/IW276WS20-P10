import argparse
import os

def replacePath(file, path):
    with open(file, "r") as f:
        data = [line.rstrip("\n") for line in f.readlines()]
        f.close()

    for i in range(len(data)):
        data[i] = os.path.join(path, os.path.basename(data[i])) + "\n"

    with open(file, "w") as f:
        f.writelines(data)
        f.close()
        print(f"Write to {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change paths of dataset images")
    parser.add_argument("-train", type=str, default="out/train/train.txt", required=True, help="Path to the file containing train images")
    parser.add_argument("-valid", type=str, default="out/train/valid.txt", required=True, help="Path to the file containing validate images")
    parser.add_argument("-path", type=str, required=True, help="New path to image files")

    args = parser.parse_args()

    replacePath(args.train, args.path)
    replacePath(args.valid, args.path)