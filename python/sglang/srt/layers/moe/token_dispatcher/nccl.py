from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)


class NcclRouteHandle(NamedTuple):
    """Source-local metadata needed to reverse one dispatch."""

    send_splits: list[int]
    recv_splits: list[int]
    send_route_idx: torch.Tensor
    num_input_tokens: int
    top_k: int


class NcclDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    route_handle: NcclRouteHandle

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL


assert isinstance(NcclDispatchOutput, DispatchOutput)


class NcclCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    route_handle: NcclRouteHandle

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.NCCL


assert isinstance(NcclCombineInput, CombineInput)


class NcclDispatcher(BaseDispatcher):
    """Eager variable-split NCCL dispatcher for static expert partitions.

    Each token/expert route is sent as one row. Only the activation, local
    expert id, and router weight cross the network; source token positions stay
    in ``NcclRouteHandle`` and are recovered from the reverse all-to-all order.
    """

    def __init__(self, group: dist.ProcessGroup, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.top_k = moe_runner_config.top_k
        num_fused_shared_experts = moe_runner_config.num_fused_shared_experts or 0

        if (
            self.num_experts is None
            or self.num_local_experts is None
            or self.top_k is None
        ):
            raise ValueError(
                "NCCL dispatcher requires global/local expert counts and top-k"
            )
        if self.top_k <= 0:
            raise ValueError("NCCL dispatcher requires top-k to be positive")
        if num_fused_shared_experts:
            raise NotImplementedError(
                "NCCL dispatcher does not support fused shared experts"
            )
        if self.num_local_experts * self.world_size != self.num_experts:
            raise ValueError(
                "NCCL dispatcher requires an equal contiguous expert partition: "
                "num_local_experts * world_size == num_experts"
            )

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        return recv_counts

    def _exchange(
        self,
        tensor: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> torch.Tensor:
        output = torch.empty(
            (sum(recv_splits), *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
        )
        return output

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> NcclDispatchOutput:
        if not hidden_states.is_cuda:
            raise NotImplementedError(
                "NCCL MoE dispatcher requires CUDA activation tensors"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NCCL MoE dispatcher does not support CUDA graph capture; "
                "disable decode and prefill CUDA graphs"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise NotImplementedError(
                "NCCL MoE dispatcher currently supports BF16 activations only"
            )
        if hidden_states.ndim != 2:
            raise ValueError(
                "NCCL MoE dispatcher expects a 2D activation tensor, got "
                f"shape={tuple(hidden_states.shape)}"
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("NCCL MoE dispatcher requires standard top-k output")

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        expected_topk_shape = (hidden_states.shape[0], self.top_k)
        if (
            topk_ids.shape != expected_topk_shape
            or topk_weights.shape != expected_topk_shape
        ):
            raise ValueError(
                "NCCL MoE dispatcher received inconsistent top-k metadata: "
                f"expected={expected_topk_shape}, ids={tuple(topk_ids.shape)}, "
                f"weights={tuple(topk_weights.shape)}"
            )

        num_tokens = hidden_states.shape[0]
        route_idx = torch.arange(
            num_tokens * self.top_k,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        token_idx = torch.div(route_idx, self.top_k, rounding_mode="floor")
        expert_idx = topk_ids.reshape(-1).to(torch.int64)
        router_weight = topk_weights.reshape(-1).to(torch.float32)

        valid = (expert_idx >= 0) & (expert_idx < self.num_experts)
        route_idx = route_idx[valid]
        token_idx = token_idx[valid]
        expert_idx = expert_idx[valid]
        router_weight = router_weight[valid]
        destination = torch.div(
            expert_idx, self.num_local_experts, rounding_mode="floor"
        )

        order = torch.argsort(destination, stable=True)
        destination = destination[order]
        send_counts = torch.bincount(destination, minlength=self.world_size).to(
            torch.int64
        )
        recv_counts = self._exchange_counts(send_counts)
        # Variable-split collectives require host lists. Copy both small count
        # vectors together so dispatch pays one device-to-host synchronization.
        send_splits, recv_splits = (
            torch.stack((send_counts, recv_counts)).cpu().tolist()
        )

        send_route_idx = route_idx[order].contiguous()
        send_expert_idx = expert_idx[order].contiguous()
        send_weight = router_weight[order].contiguous()
        send_hidden_states = hidden_states[token_idx[order]].contiguous()

        recv_hidden_states = self._exchange(
            send_hidden_states, send_splits, recv_splits
        )
        recv_expert_idx = self._exchange(send_expert_idx, send_splits, recv_splits)
        recv_weight = self._exchange(send_weight, send_splits, recv_splits)

        local_topk_output = StandardTopKOutput(
            topk_weights=recv_weight.unsqueeze(1),
            topk_ids=torch.remainder(recv_expert_idx, self.num_local_experts)
            .to(torch.int32)
            .unsqueeze(1),
            router_logits=None,
        )
        return NcclDispatchOutput(
            hidden_states=recv_hidden_states,
            topk_output=local_topk_output,
            route_handle=NcclRouteHandle(
                send_splits=send_splits,
                recv_splits=recv_splits,
                send_route_idx=send_route_idx,
                num_input_tokens=num_tokens,
                top_k=self.top_k,
            ),
        )

    def combine(self, combine_input: NcclCombineInput) -> torch.Tensor:
        hidden_states, handle = combine_input
        expected_records = sum(handle.recv_splits)
        if hidden_states.shape[0] != expected_records:
            raise RuntimeError(
                "NCCL MoE runner changed the dispatched record count: "
                f"output={hidden_states.shape[0]}, expected={expected_records}"
            )

        returned_hidden_states = self._exchange(
            hidden_states,
            handle.recv_splits,
            handle.send_splits,
        )
        return self._deterministic_combine(
            returned_hidden_states,
            handle.send_route_idx,
            handle.num_input_tokens,
            handle.top_k,
        )

    @staticmethod
    def _deterministic_combine(
        hidden_states: torch.Tensor,
        route_indices: torch.Tensor,
        num_input_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        """Accumulate routes in canonical top-k slot order without atomics."""
        if hidden_states.ndim != 2:
            raise ValueError(
                "NCCL combine expects a 2D activation tensor, got "
                f"shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != route_indices.numel():
            raise RuntimeError(
                "NCCL combine route-index count does not match returned rows: "
                f"rows={hidden_states.shape[0]}, routes={route_indices.numel()}"
            )
        if top_k <= 0:
            raise ValueError("NCCL combine requires top-k to be positive")

        combined = torch.zeros(
            (num_input_tokens, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for slot in range(top_k):
            slot_mask = torch.remainder(route_indices, top_k) == slot
            slot_routes = route_indices[slot_mask]
            slot_tokens = torch.div(slot_routes, top_k, rounding_mode="floor")
            updates = combined.index_select(0, slot_tokens) + hidden_states[slot_mask]
            combined.index_copy_(0, slot_tokens, updates)
        return combined
