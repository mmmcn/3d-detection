from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, Extension


if __name__ == '__main__':

    setup(
        name='manhattan_weights_farthest_point_sample',
        ext_modules=[
            CUDAExtension(
                name='manhattan_weights_fps_ext',
                sources=['src/manhattan_weights_farthest_point_sample.cpp',
                         'src/manhattan_weights_farthest_point_sample_cuda.cu']
            )],
        cmdclass={'build_ext': BuildExtension})
