import sys
import os
import random

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("usage: splitInTrainAndValidate.py ANNOTATIONS_FILE PERCENT_TRAIN(FLOAT)")
    
    else:
        valPath = os.path.join(os.path.dirname(sys.argv[1]), "val.txt")
        trainPath = os.path.join(os.path.dirname(sys.argv[1]), "train.txt")

        valAmount = 0
        trainAmount = 0

        with open(sys.argv[1], "r") as file, open(valPath, "w+") as val, open(trainPath, "w+") as train:
            for line in file:
                rand = random.random()
                if rand >= 0 and rand <= float(sys.argv[2]) / 100:
                    train.write(line)
                    trainAmount += 1
                else:
                    val.write(line)
                    valAmount += 1

        print(f"Saved splits (Training: {float(sys.argv[2])}%):\nTrain: {trainPath} Amount: {trainAmount}\nValidate:{valPath} Amount: {valAmount}")
