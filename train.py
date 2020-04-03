import argparse
from trainers.controller import Controller

# Argument parser
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True,
                    help="configuration of the experiment")
parser.add_argument("--setup", type=str, required=True)
parser.add_argument("--result_dir", type=str, required=False, default='Results')
args = parser.parse_args()

# Create controller
controller = Controller(cfg_path=args.cfg, result_dir=args.result_dir,
                        setup=args.setup)