#!/usr/local/bin/python
#
#  printf.py
#  $Id: printf.py,v 1.1 2018/03/07 12:48:02 daichi Exp $
#
import sys

def eprint (s):
    sys.stdout.write ('\r%s' % s)
    sys.stdout.flush ()

