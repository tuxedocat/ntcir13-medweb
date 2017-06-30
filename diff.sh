#!/bin/bash
diff -U 0 reports/suf-bow/analysis.csv reports/suf-bow_pas/analysis.csv | dwdiff -C0 --diff-input -d, | colordiff --difftype=wdiff | less -R
