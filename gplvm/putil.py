#!/bin/env python
#
#    putil.py
#    Matplot plotting utilities.
#    $Id: putil.py,v 1.8 2018/03/07 12:48:13 daichi Exp $
#
from pylab import *

# set aspect ratio.
def aspect_ratio (r):
    gca().set_aspect (r)

# set font specification.
def setfonts(spec):
    matplotlib.rc('font', **spec)

# set default line widths.
def linewidth(n):
    matplotlib.rcParams['axes.linewidth'] = n

# 'nomirror' in Gnuplot.
def nomirror():
    ax = gca()
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

# axes lie on zeros.
def zero_origin():
    ax = axes()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
# leave only left and bottom axis.
# eg: putil.simpleaxis()
def simpleaxis():
    ax = gca().axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# one dimensional plot.
def one_dimensional():
    axes().spines['left'].set_visible(False)
    tick_params (
        left='off',
        labelleft='off'
    )

# add 'x' and 'y'.    
def add_xy ():
    ax = gca().axes
    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]
    ax.text(0,xmax+0.1,r'$y$',ha='center')
    ax.text(ymax+0.1,0,r'$x$',va='center')

# set margins outside of labels.
# eg: putil.margins(left=0.1,bottom=0.2)
margins = matplotlib.pyplot.subplots_adjust

#
# Ticks
#

def no_ticks ():
    ax = axes()
    ax.get_xaxis().set_ticks ([])
    ax.get_yaxis().set_ticks ([])

# padding of xticks and yticks.
def tickpad(n):
    axes().tick_params(direction='out',pad=n)
def xtickpad(n):
    gca().get_xaxis().set_tick_params(direction='out',pad=n)
def ytickpad(n):
    gca().get_yaxis().set_tick_params(direction='out',pad=n)

# xtick and ytick labels.
# usage: xticklabels(("foo","bar"))
def xticklabels(s):
    gca().set_xticklabels(s)
def yticklabels(s):
    gca().set_yticklabels(s)

# set ticks size.
# usage: ticksize(10,1)
def ticksize(length,width):
    for line in gca().get_xticklines() + gca().get_yticklines():
        line.set_markersize(length)
        line.set_markeredgewidth(width)

def savefig (file):
    plt.savefig (file, bbox_inches='tight')
