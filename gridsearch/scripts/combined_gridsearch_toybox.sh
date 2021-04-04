#!/bin/bash
#GPU0
sh ./scripts/gridsearches_toybox/EWC.sh
wait
sh ./scripts/gridsearches_toybox/L2.sh
wait
sh ./scripts/gridsearches_toybox/MAS.sh
wait
sh ./scripts/gridsearches_toybox/SI.sh
wait
sh ./scripts/gridsearches_toybox/iCARL.sh
wait

#GPU1
sh ./scripts/gridsearches_toybox/NaiveRehearsal.sh
wait
sh ./scripts/gridsearches_toybox/GEM.sh
wait
sh ./scripts/gridsearches_toybox/NormalNNlower.sh
wait
sh ./scripts/gridsearches_toybox/lower_bound.sh
wait
