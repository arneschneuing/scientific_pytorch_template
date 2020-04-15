import argparse

from src.trainers.controller import Controller

# Argument parser
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True,
                    help="configuration of the experiment")
parser.add_argument("--session", type=str, required=True, help="Session name")
parser.add_argument("--result_dir", type=str, required=False,
                    default='results')
parser.add_argument("--result_filename", type=str, required=False,
                    default='result_overview.csv')
parser.add_argument("--overwrite",
                    help="overwrite session directory with new data",
                    action="store_true")
args = parser.parse_args()

# Create controller
controller = Controller(cfg_path=args.cfg, result_dir=args.result_dir,
                        session=args.session, overwrite=args.overwrite,
                        result_filename=args.result_filename)

# Start main loop
controller.start()
