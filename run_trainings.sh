#!/bin/bash
set -e 

for filename in cleaned_data/*.txt; do
    [ -e "$filename" ] || continue
    echo $filename
    ./demo.sh "$filename"
done

