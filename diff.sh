#!/bin/bash
diff -U 0 $1/analysis.csv $2/analysis.csv | dwdiff -C0 --diff-input -d, | colordiff --difftype=wdiff | less -R
