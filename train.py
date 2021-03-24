import argparse

from src.core_components.controller import Controller
from src.utilities.template_utils import boolean_string

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
parser.add_argument("--interactive", type=boolean_string, required=False,
                    default=True, help="if False, skip all user inputs and "
                                       "select default actions")
args = parser.parse_args()

# Create controller
controller = Controller(cfg_path=args.cfg, result_dir=args.result_dir,
                        session=args.session, overwrite=args.overwrite,
                        result_filename=args.result_filename,
                        interactive=args.interactive)

# Start main loop
controller.start()
