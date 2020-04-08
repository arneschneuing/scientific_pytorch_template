import argparse
from src.trainers.controller import Controller

# Argument parser
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True,
                    help="configuration of the experiment")
parser.add_argument("--setup", type=str, required=True)
parser.add_argument("--result_dir", type=str, required=False,
                    default='results')
parser.add_argument("--overwrite",
                    help="overwrite setup directory with new data",
                    action="store_true")
args = parser.parse_args()

# Create controller
controller = Controller(cfg_path=args.cfg, result_dir=args.result_dir,
                        setup=args.setup, overwrite=args.overwrite)

# Start main loop
controller.start()
