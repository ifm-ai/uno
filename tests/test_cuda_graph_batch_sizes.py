import unittest

from nano_vllm_uno.engine.block_cuda_graph_runner import BlockCudaGraphRunner


class CudaGraphBatchSizeTest(unittest.TestCase):
    def test_speculative_ladder_through_64(self):
        self.assertEqual(
            BlockCudaGraphRunner.capture_batch_sizes(64),
            list(range(1, 9))
            + list(range(10, 33, 2))
            + list(range(40, 65, 4)),
        )

    def test_exact_non_bucket_maximum_is_included(self):
        self.assertEqual(
            BlockCudaGraphRunner.capture_batch_sizes(9),
            list(range(1, 10)),
        )
        self.assertEqual(
            BlockCudaGraphRunner.capture_batch_sizes(300)[-3:],
            [272, 288, 300],
        )

    def test_ladder_has_no_legacy_512_limit(self):
        sizes = BlockCudaGraphRunner.capture_batch_sizes(544)
        self.assertEqual(sizes[-3:], [512, 528, 544])

    def test_explicit_batch_sizes_bypass_default_ladder(self):
        self.assertEqual(
            BlockCudaGraphRunner.resolve_capture_batch_sizes(
                64,
                [1, 7, 64],
            ),
            [1, 7, 64],
        )


if __name__ == "__main__":
    unittest.main()
