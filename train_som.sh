WIDTH=$1
HEIGHT=$2
SUF=\_h"$HEIGHT"\_w"$WIDTH"\_vlass
IMG=$3

# PINKDIR='/scratch1/gal16b/Circular_PINK/PINK/test_install_path/bin'

#module load pink/2.4
# module load cuda/10.1.168

OUT1="SOM_B1$SUF.bin"
if [[ ! -f $OUT1 ]]; then
    Pink --train $IMG $OUT1 --numthreads 10 --som-width $WIDTH --som-height $HEIGHT --num-iter 5  --dist-func unitygaussian 2.5 0.1 --init random --inter-store keep -n 180 -p 10 --euclidean-distance-shape circular 
fi

OUT2="SOM_B2$SUF.bin"
if [[ ! -f $OUT2 ]]; then
    Pink --train $IMG $OUT2 --init $OUT1 --numthreads 10 --som-width $WIDTH --som-height $HEIGHT --num-iter 5 --dist-func unitygaussian 1.5 0.05 --inter-store keep -n 180 -p 10 --euclidean-distance-shape circular
fi

OUT3="SOM_B3$SUF.bin"
if [[ ! -f $OUT3 ]]; then
    Pink --train $IMG $OUT3 --init $OUT2 --numthreads 10 --som-width $WIDTH --som-height $HEIGHT --num-iter 10 --dist-func unitygaussian 0.7 0.05  --inter-store keep -p 10 -n 360 --euclidean-distance-shape circular
fi


# Pink --map $IMG MAP_B3$SUF.bin $OUT3 --numthreads 4 --som-width $WIDTH --som-height $HEIGHT --store-rot-flip TRANSFORM_B3$SUF.bin --euclidean-distance-shape circular