#!/bin/bash
#GPU0
#sh ./scripts/optimal_toybox/EWC.sh toybox 1
#wait
#sh ./scripts/optimal_toybox/L2.sh toybox 1
#wait
#sh ./scripts/optimal_toybox/MAS.sh toybox 1
#wait
sh ./scripts/optimal_toybox/SI.sh toybox 3
wait
sh ./scripts/optimal_toybox/iCARL.sh toybox 3
wait
