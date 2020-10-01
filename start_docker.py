import os
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use this script to start the docker image")
    parser.add_argument("--input", help="The directory, which contains all test images")
    parser.add_argument("--output", help="Directory where the output json (COCO Format) should be stored")
    parser.add_argument("--image", help="Name of the docker image", nargs='?', default="iw276ws20-p10:0.1")
    parser.add_argument("--valid_json",
                        help="Name of the valid coco json file (the file has to be a direct member of the input "
                             "directory)")
    parser.add_argument("--tiny", help="Use the TINY YOLOv4 model to infer the images", action="store_true")
    parser.add_argument('--write_images', action="store_true",
                        help='Write images with detected bounding boxes to output directory')
    args = parser.parse_args()

    if not os.path.exists(args.input) or not os.path.exists(args.output):
        raise ValueError("The specified in- or output directory does not exist!")

    weights = "custom-yolov4-detector_final-416x416"
    if args.tiny:
        weights = "custom-yolov4-tiny-detector_final-416x416"

    writeImages = "--write_images" if args.write_images else ""
    print(f"DEBUG: STARTING IMAGE {args.image}")
    print(f"DEBUG: USING WEIGHTS {weights}")
    if args.write_images:
        print(f"DEBUG: WRITING IMAGES TO {args.output}/images")

    weights = os.path.join("/home/IW276WS20-P10/pretrained-models", weights + ".trt")

    subprocess.run(filter(lambda x: x != "", [
        "sudo",
        "docker",
        "run",
        "-it",
        "--rm",
        "--runtime",
        "nvidia",
        "--network",
        "host",
        "-v",
        f"{args.input}:/home/in",
        "-v",
        f"{args.output}:/home/out",
        args.image,
        "python3",
        "trt_yolo.py",
        "--imageDir",
        "/home/in/images",
        "-m",
        weights,
        "-v",
        f"/home/in/{os.path.basename(args.valid_json)}",
        writeImages
    ]))
