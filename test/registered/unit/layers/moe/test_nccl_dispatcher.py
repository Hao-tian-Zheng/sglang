"""Multi-GPU correctness tests for the eager NCCL MoE dispatcher."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.nccl import (
    NcclCombineInput,
    NcclDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="2-gpu-large")


class TestNcclDispatcher(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _dispatcher(self, hidden_size: int, top_k: int = 2) -> NcclDispatcher:
        return NcclDispatcher(
            group=dist.group.WORLD,
            moe_runner_config=MoeRunnerConfig(
                num_experts=self.world_size * 4,
                num_local_experts=4,
                num_fused_shared_experts=0,
                hidden_size=hidden_size,
                top_k=top_k,
                params_dtype=torch.bfloat16,
            ),
        )

    def test_uneven_routes_round_trip(self):
        num_tokens, hidden_size, top_k = 257, 128, 2
        token = torch.arange(num_tokens, device="cuda")
        feature = torch.arange(hidden_size, device="cuda")
        hidden_states = (
            self.rank + token[:, None] / 512.0 + feature[None, :] / 4096.0
        ).to(torch.bfloat16)

        destinations = torch.stack(
            (
                torch.where(
                    token % 10 < 7,
                    torch.zeros_like(token),
                    (self.rank + token) % self.world_size,
                ),
                (self.rank + token + 1) % self.world_size,
            ),
            dim=1,
        )
        local_experts = torch.stack(
            tuple((token + self.rank + slot) % 4 for slot in range(top_k)),
            dim=1,
        )
        topk_ids = (destinations * 4 + local_experts).to(torch.int32)
        topk_weights = torch.tensor(
            (2.0 / 3.0, 1.0 / 3.0), dtype=torch.float32, device="cuda"
        ).expand(num_tokens, -1)

        dispatcher = self._dispatcher(hidden_size)
        dispatched = dispatcher.dispatch(
            hidden_states,
            StandardTopKOutput(topk_weights, topk_ids, None),
        )
        local_ids = dispatched.topk_output.topk_ids[:, 0]
        self.assertTrue(torch.all((local_ids >= 0) & (local_ids < 4)).item())

        scales = 1.0 + local_ids.float() / 8.0
        local_output = (
            dispatched.hidden_states.float()
            * dispatched.topk_output.topk_weights[:, 0, None]
            * scales[:, None]
        )
        combined = dispatcher.combine(
            NcclCombineInput(local_output, dispatched.route_handle)
        )

        reference = torch.zeros_like(combined)
        for slot in range(top_k):
            scale = 1.0 + local_experts[:, slot].float() / 8.0
            reference += (
                hidden_states.float() * topk_weights[:, slot, None] * scale[:, None]
            )
        torch.testing.assert_close(combined, reference, rtol=1e-5, atol=1e-5)

    def test_empty_source_rank(self):
        hidden_size = 16
        num_tokens = 0 if self.rank == 1 else 8
        hidden_states = torch.full(
            (num_tokens, hidden_size),
            self.rank + 1.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        target = (self.rank + 1) % self.world_size
        topk_ids = torch.full(
            (num_tokens, 1), target * 4, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device="cuda")

        dispatcher = self._dispatcher(hidden_size, top_k=1)
        dispatched = dispatcher.dispatch(
            hidden_states,
            StandardTopKOutput(topk_weights, topk_ids, None),
        )
        combined = dispatcher.combine(
            NcclCombineInput(dispatched.hidden_states.float(), dispatched.route_handle)
        )
        self.assertEqual(combined.shape, (num_tokens, hidden_size))
        if num_tokens:
            torch.testing.assert_close(combined, hidden_states.float(), rtol=0, atol=0)

    def test_all_ranks_empty(self):
        hidden_states = torch.empty((0, 16), dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.empty((0, 1), dtype=torch.int32, device="cuda")
        topk_weights = torch.empty((0, 1), dtype=torch.float32, device="cuda")
        dispatcher = self._dispatcher(hidden_size=16, top_k=1)

        dispatched = dispatcher.dispatch(
            hidden_states,
            StandardTopKOutput(topk_weights, topk_ids, None),
        )
        combined = dispatcher.combine(
            NcclCombineInput(dispatched.hidden_states, dispatched.route_handle)
        )
        self.assertEqual(combined.shape, (0, 16))

    def test_combine_is_independent_of_physical_route_order(self):
        num_tokens, hidden_size, top_k = 257, 32, 8
        num_routes = num_tokens * top_k
        hidden_states = torch.randn(
            (num_routes, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        route_indices = torch.arange(num_routes, dtype=torch.int64, device="cuda")

        expected = NcclDispatcher._deterministic_combine(
            hidden_states, route_indices, num_tokens, top_k
        )
        permutation = torch.randperm(num_routes, device="cuda")
        remapped = NcclDispatcher._deterministic_combine(
            hidden_states.index_select(0, permutation),
            route_indices.index_select(0, permutation),
            num_tokens,
            top_k,
        )
        torch.testing.assert_close(expected, remapped, rtol=0, atol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
