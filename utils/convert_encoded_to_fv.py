#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script reads in the encoded feature vectors and the targets, and
creates a .

The code is developed and teset with Python 3.7, scikit-learn 0.24.2
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv, re, json

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2020~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2021-07-10'

def process_fv(fvfile, eventfile):
    # Read all events
    events = pd.read_csv(eventfile, index_col=0)
    # Read feature vector
    tmp = pd.read_csv(fvfile)
    dates = list(tmp.timestamp)
    dates = [int(d.replace('.','')) for d in dates]     # Convert YYYY.MM.dd to YYYYMMdd
    tmp.index = dates
    # Merge events and feature vectors with index
    fv = tmp.loc[events.index, np.arange(2048).astype('str')]
    return(fv)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Evaluate the performance of GLM with various feature vectors and events.')
    parser.add_argument('--encoded_file', '-i', help='the file containing encoded vectors.')
    parser.add_argument('--reference_file', '-r', help='the reference file (events).')
    parser.add_argument('--output', '-o', default='fv.csv', help='the output files containing tidy feature vectors.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Perform conversion
    fv = process_fv(args.encoded_file, args.reference_file)
    # Write output
    fv.to_csv(args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()