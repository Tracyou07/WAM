"""Bounded all-reduce for replicated parameters; FSDP shards are already reduced."""
from collections.abc import Iterable
import torch
import torch.distributed as dist
from torch import nn

_GRADIENT_BUCKET_ELEMENTS = 4_000_000


def synchronize_replicated_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    skip_parameter_ids: set[int],
    world_size: int,
) -> None:
    from torch.distributed.tensor import DTensor

    selected = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and id(parameter) not in skip_parameter_ids
        and not isinstance(parameter, DTensor)
    ]
    if not selected or world_size <= 1:
        return
    device = selected[0].device
    presence = torch.tensor(
        [parameter.grad is not None for parameter in selected],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    active_parameters = [
        parameter for parameter, active in zip(selected, presence.tolist(), strict=True)
        if active > 0
    ]
    for parameter in active_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        elif not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()

    views: list[torch.Tensor] = []
    references: list[tuple[nn.Parameter, int, int]] = []
    bucket_numel = 0

    def reduce_bucket() -> None:
        nonlocal bucket_numel
        if not views:
            return
        bucket = torch.cat(views)
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
        bucket.div_(world_size)
        cursor = 0
        for parameter, start, end in references:
            length = end - start
            parameter.grad.view(-1)[start:end].copy_(bucket[cursor : cursor + length])
            cursor += length
        views.clear()
        references.clear()
        bucket_numel = 0

    for parameter in active_parameters:
        flat = parameter.grad.view(-1)
        start = 0
        while start < flat.numel():
            take = min(_GRADIENT_BUCKET_ELEMENTS - bucket_numel, flat.numel() - start)
            end = start + take
            views.append(flat[start:end])
            references.append((parameter, start, end))
            bucket_numel += take
            start = end
            if bucket_numel == _GRADIENT_BUCKET_ELEMENTS:
                reduce_bucket()
    reduce_bucket()
