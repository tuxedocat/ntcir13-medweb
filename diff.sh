#!/bin/bash
diff -U 0 reports/suf/analysis.csv reports/suf+pas/analysis.csv | dwdiff -C0 --diff-input -d, | colordiff --difftype=wdiff | less -R
