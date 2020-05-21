#!/usr/bin/env python
# coding=utf-8

from subprocess import call
from os import listdir, remove
from os.path import abspath, dirname

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--key', nargs='+', default=[], type=str)
parser.add_argument('--no-key', nargs='+', default=[], type=str)
args = parser.parse_args()

fnames = [fn for fn in listdir(dirname(abspath(__file__)))
          if '.out' in fn and '-' in fn and all([k in fn for k in args.key])
          and all([k not in fn for k in args.no_key])]
ids = [fn.strip().strip('.out').split('-')[1] for fn in fnames]
for idx in ids:
    call(['scancel', idx])
for fn in fnames:
    remove(fn)
