OPSPATH="lib/utils/cuda_ops"

cd ${OPSPATH}

# density_and_manhattan_weights_meanwhile_fps
# only need density_and_manhattan_weights_meanwhile_fps
cd density_and_manhattan_weights_meanwhile_fps
python setup.py build

ext_file=`ls build/ | grep lib`
ext_file=$ext_file/`ls build/$ext_file`
cp build/$ext_file .

rm -rf build

cd ../../../../
