import unittest

from nano_vllm_uno.jit_kernel.qk_norm import fused_qk_rms_norm
from nano_vllm_uno.layers.attention import (
    _call_with_kvcache,
    store_kvcache,
)


class CompileCustomOpsTest(unittest.TestCase):
    def test_flash_kvcache_is_a_compile_boundary(self):
        self.assertTrue(_call_with_kvcache._torchdynamo_disable)

    def test_mutating_custom_kernels_are_compile_boundaries(self):
        self.assertTrue(fused_qk_rms_norm._torchdynamo_disable)
        self.assertTrue(store_kvcache._torchdynamo_disable)


if __name__ == "__main__":
    unittest.main()
