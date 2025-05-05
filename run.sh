#/bin/bash

for i in {1..4}; do
  echo "running run $i"
  python test_petric.py --outdir output_$i
done
