import os
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Split a given dataset into training and validation images")
    parser.add_argument("-path", required=True, type=str, help="Path to annotations file or the image folder (when "
                                                               "using the -darknet option)")
    parser.add_argument("-output", required=True, type=str, help="Output directory")
    parser.add_argument("-split", required=True, type=float, default=80, help="Amount of training data in percent, "
                                                                              "e.g. 80")
    parser.add_argument("-darknet", type=bool, help="Should the output be compatible with the darknet required format?")

    args = parser.parse_args()

    darknet = args.darknet

    if not os.path.exists(args.output): os.mkdir(args.output)

    valPath = os.path.join(args.output, "val.txt")
    trainPath = os.path.join(args.output, "train.txt")

    valAmount = 0
    trainAmount = 0

    with open(valPath, "w+") as val, open(trainPath, "w+") as train:
        if darknet:
            for file in os.listdir(args.path):
                rand = random.random()
                if 0 <= rand <= float(args.split) / 100:
                    train.write(os.path.basename(file) + "\n")
                    trainAmount += 1
                else:
                    val.write(os.path.basename(file) + "\n")
                    valAmount += 1
        else:
            with open(args.path, "r") as file:
                for line in file:
                    rand = random.random()
                    if 0 <= rand <= float(args.split) / 100:
                        train.write(line)
                        trainAmount += 1
                    else:
                        val.write(line)
                        valAmount += 1

    print(f"Saved splits (Training: {float(args.split)}%):\nTrain: {trainPath} Amount: {trainAmount}\nValidate:{valPath} Amount: {valAmount}")
