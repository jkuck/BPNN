#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys, subprocess as sp
os.chdir(osp.dirname(osp.dirname(osp.abspath(__file__))))
sp.call(['bash', './batch/'+sys.argv[1]])
