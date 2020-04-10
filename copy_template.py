import os
import argparse
from src.utilities.util import safe_copytree

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("dest", type=str, nargs='?', default='../new_project')
args = parser.parse_args()
destination = args.dest

# Get current directory
template_dir = os.path.dirname(os.path.realpath(__file__))

# Make sure the right folder is copied
assert os.path.basename(template_dir) == 'scientific_pytorch_template'

# Read gitignore
gitignore = os.path.join(template_dir, '.gitignore')
with open(gitignore) as f:
    ignore = f.readlines()
ignore = [x.strip('\n').strip('/') for x in ignore]

# Add further items to ignore
ignore += ['.gitignore', '.git', 'copy_template.py', 'README.md', 'images']

# Copy files to new location
safe_copytree(template_dir, destination, ignore)
print(f'Copied template from {template_dir} to {os.path.abspath(destination)}')
