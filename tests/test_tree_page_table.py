import unittest

import torch

from nano_vllm_uno.engine.tree_page_table import build_tree_page_table


class TreePageTableTests(unittest.TestCase):

    def _run_case(self, device: torch.device) -> None:
        parents = torch.tensor(
            [[-1, 0, 0, 1], [-1, 0, 1, 0]],
            dtype=torch.long,
            device=device,
        )
        depths = torch.tensor(
            [[0, 1, 1, 2], [0, 1, 2, 1]],
            dtype=torch.long,
            device=device,
        )
        block_tables = torch.tensor(
            [[3], [5]],
            dtype=torch.int32,
            device=device,
        )
        kv_seqlens = torch.tensor([104, 204], dtype=torch.int32, device=device)
        prefix = torch.empty(2, dtype=torch.int32, device=device)
        positions = torch.empty(8, dtype=torch.long, device=device)
        slots = torch.empty(8, dtype=torch.int32, device=device)
        page_table = torch.full(
            (8, 4),
            -1,
            dtype=torch.int32,
            device=device,
        )
        cache_seqlens = torch.empty(8, dtype=torch.int32, device=device)

        build_tree_page_table(
            parents,
            depths,
            block_tables,
            kv_seqlens,
            prefix,
            positions,
            slots,
            page_table,
            cache_seqlens,
            256,
        )

        torch.testing.assert_close(
            prefix.cpu(),
            torch.tensor([100, 200], dtype=torch.int32),
        )
        torch.testing.assert_close(
            cache_seqlens.cpu(),
            torch.tensor([1, 2, 2, 3, 1, 2, 3, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            positions.cpu(),
            torch.tensor([100, 101, 101, 102, 200, 201, 202, 201]),
        )
        torch.testing.assert_close(
            slots.cpu(),
            torch.tensor(
                [868, 869, 870, 871, 1480, 1481, 1482, 1483],
                dtype=torch.int32,
            ),
        )
        expected = [
            [868],
            [868, 869],
            [868, 870],
            [868, 869, 871],
            [1480],
            [1480, 1481],
            [1480, 1481, 1482],
            [1480, 1483],
        ]
        for row, values in enumerate(expected):
            self.assertEqual(
                page_table[row, : len(values)].cpu().tolist(),
                values,
            )

    def test_cpu_reference(self):
        self._run_case(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_triton_kernel(self):
        self._run_case(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
