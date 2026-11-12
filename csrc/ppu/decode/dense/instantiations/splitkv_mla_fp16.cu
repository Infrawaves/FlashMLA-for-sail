#ifndef FLASH_MLA_DISABLE_FP16
#include "decode/dense/splitkv_mla.h"

template void run_flash_splitkv_mla_kernel<cutlass::half_t, 80>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_flash_splitkv_mla_kernel<cutlass::half_t, 89>(Flash_fwd_mla_params &params, cudaStream_t stream);
#endif
