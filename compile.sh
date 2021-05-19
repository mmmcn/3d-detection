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

# compile mmdet3d ops

MMDET3D_OPS_PATH="mmdet3d/ops"
cd ${MMDET3D_OPS_PATH}

# ball_query
cd ball_query
python setup.py build
../run.sh
cd ..

# furthest_point_sample
cd furthest_point_sample
python setup.py build
../run.sh
cd ..


# gather_points
cd gather_points
python setup.py build
../run.sh
cd ..

# group_points
cd group_points
python setup.py build
../run.sh
cd ..

# interpolate
cd interpolate
python setup.py build
../run.sh
cd ..

# iou3d
cd iou3d
python setup.py build
../run.sh
cd ..

# roiaware_pool3d
cd roiaware_pool3d
python setup.py build
../run.sh

cd ../../../

