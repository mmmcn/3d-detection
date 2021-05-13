OPSPATH="lib/utils/cuda_ops"


# density_and_manhattan_weights_fps
cd ${OPSPATH}/density_and_manhattan_weights_fps
./build_and_clean.sh

# density_and_manhattan_weights_meanwhile_fps
cd density_and_manhattan_weights_meanwhile_fps
./build_and_clean.sh

# density_weights_fps
cd density_weights_fps
./build_and_clean.sh


# manhattan_weights_fps
cd manhattan_weights_fps
./build_and_clean.sh

# voxel_ball_query
cd voxel_ball_query
./build_and_clean.sh


# voxel_generator
cd voxel_generator
./build_and_clean.sh

# voxel_group_points
cd voxel_group_points
./build_and_clean.sh
