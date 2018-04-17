# -*- coding: utf-8 -*-
import sys
import h5py

# Print iterations progress
# Credit: aubricus
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
# Updated: edding
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    if filled_length == 0 or filled_length == bar_length:
        bar = '=' * filled_length + '.' * (bar_length - filled_length)
    else:
        bar = '=' * (filled_length - 1) + '>' + '.' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()