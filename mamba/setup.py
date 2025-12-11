import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from pathlib import Path

# 判断是否是调试模式
def append_nvcc_threads(extra_flags):
    # 可根据需要在此添加更多编译标志
    return extra_flags

# 设置编译参数
cc_flag = []

# 获取当前路径
this_dir = Path(__file__).parent

# 设置 CUDA 扩展
ext_modules = [
    CUDAExtension(
        name="select_scan_cuda",
        sources=[
            "csrc/selective_scan/selective_scan.cpp",
            "csrc/selective_scan/selective_scan_fwd_fp32.cu",
            "csrc/selective_scan/selective_scan_fwd_fp16.cu",
            "csrc/selective_scan/selective_scan_fwd_bf16.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O2",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
            ),
        },
        include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
    )
]

setup(
    name="select_scan_cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
    language="c++",
)
