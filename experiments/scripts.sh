#!/bin/bash

for lhs in 0.9 0.95 0.99
do
	echo "Running" ${lhs} gamma 
	python table5_pwm.py ${lhs} gamma &
	echo "Running" ${lhs} lognorm
	python table5_pwm.py ${lhs} lognorm &
	echo "Running" ${lhs} pareto
	python table5_pwm.py ${lhs} pareto &
done
wait 
echo "done all processes."