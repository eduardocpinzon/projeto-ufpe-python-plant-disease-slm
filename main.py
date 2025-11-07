"""
Plant Disease Classifier
Usage: python main.py [train|predict|stats] [args]
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN = ROOT / "trainingModel"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    os.chdir(TRAIN)

    if cmd == "train":
        import plant_disease_classification
        plant_disease_classification.main()

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image>")
            sys.exit(1)
        import predict
        sys.argv = ['predict.py', sys.argv[2]]
        predict.main()

    elif cmd == "stats":
        dataset = ROOT / "New Plant Diseases Dataset/train"
        classes = sorted([d.name for d in dataset.iterdir() if d.is_dir() and not d.name.startswith('.')])
        count = sum(len(list((dataset/c).iterdir())) for c in classes)
        print(f"\n{len(classes)} classes | {count:,} images\n")

    else:
        print(f"Unknown: {cmd}\n" + __doc__)

