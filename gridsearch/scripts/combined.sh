#!/bin/bash
sh ./scripts/regularization/EWC.sh
wait
sh ./scripts/iCARL.sh
wait
sh ./scripts/NaiveRehearsal.sh
wait
sh ./scripts/GEM.sh
wait
sh ./scripts/regularization/L2.sh
wait
sh ./scripts/regularization/MAS.sh
wait
sh ./scripts/regularization/SI.sh
wait
sh ./scripts/upper_bound.sh
wait
sh ./scripts/lower_bound.sh
wait
