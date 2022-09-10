#!/bin/bash
### use Python 3.8.8 in Xinyu's machine. Recommend to use python3. minimum version requirement for Python is not known yet.
echo "Start to run all four tasks."

echo "Run on tail probability estimation problem. "

echo "Warming up. Run a small example."
python tailProbabilityEstimationUnit.py

echo "Run on tail probability estimation--single threshold."
python tailProbabilityEstimationSingleThreshold.py

echo "Run on tail probability estimation--multiple thresholds."
python tailProbabilityEstimationMultipleThresholds.py

echo "Run on tail quantile estimation problem. "
echo "Warming up. Run a small example."
python quantileEstimationUnit.py

echo "Run on quantile estimation--single threshold."
python quantileEstimationSingleThreshold.py

echo "Run on quantile estimation--multiple thresholds."
python quantileEstimationMultipleThresholds.py

