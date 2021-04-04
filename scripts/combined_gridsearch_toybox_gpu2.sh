#GPU1
sh ./scripts/gridsearches_toybox/NaiveRehearsal.sh
wait
sh ./scripts/gridsearches_toybox/GEM.sh
wait
sh ./scripts/gridsearches_toybox/NormalNNlower.sh
wait
sh ./scripts/gridsearches_toybox/NormalNNupper.sh
wait
