sequence=S1_Box
total=`ls -lah ../Data/HEVA_Validate/$sequence'_1_C1'/Image/*.png | wc -l`
total=$(( total-1 ))
 
executable=~/VirtualEnvs/tensorflow_13/bin/python 

for ((idx=0; idx<=total; idx++))
do
	echo $idx
	$executable frame_fit.py $sequence $idx
done
