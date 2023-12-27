"""Communication manager for lightning attention data parallel + sequence parallel."""

import io
import torch
import torch.distributed as dist
from .seq_parallel_utils import GlobalMemoryBuffer
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# Sequence parallel group, world size, rank
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# This group includes processes for both data and sequence parallelisms.
# We use this group to reduce gradients and shard parameters and optimizer stages for ZeRO.
_SEQUENCE_DATA_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
_SEQUENCE_DATA_PARALLEL_RANK = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def initialize_lightning_seq_parallel(
    data_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
) -> None:
    """Initialize parallel groups."""
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    enable_lightning_sequence_parallel = sequence_parallel_size > 1
    if enable_lightning_sequence_parallel:
        if world_size % sequence_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})"
            )

    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

    num_data_parallel_groups: int = world_size // data_parallel_size
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks = []

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, \
        'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    # Build the sequence data parallel groups.
    global _SEQUENCE_DATA_PARALLEL_GROUP
    assert _SEQUENCE_DATA_PARALLEL_GROUP is None, \
        'sequence data parallel group is already initialized'
    all_data_sequence_parallel_group_ranks = []
    if enable_lightning_sequence_parallel:
        for i in range(num_sequence_data_parallel_groups):
            ranks = range(i * sequence_data_parallel_size,
                        (i + 1) * sequence_data_parallel_size)
            group = torch.distributed.new_group(ranks)
            all_data_sequence_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                _SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        _SEQUENCE_DATA_PARALLEL_GROUP = _DATA_PARALLEL_GROUP

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None

def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_data_parallel_is_initialized():
    """Check if sequence data parallel groups are initialized."""
    if _SEQUENCE_DATA_PARALLEL_GROUP is None:
        return False
    return True

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, \
        'sequence data parallel group is not initialized'
    return _SEQUENCE_DATA_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def set_sequence_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = world_size

def set_sequence_data_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    _SEQUENCE_DATA_PARALLEL_WORLD_SIZE = world_size

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def get_sequence_data_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    if _SEQUENCE_DATA_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())


def set_sequence_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_PARALLEL_RANK
    _SEQUENCE_PARALLEL_RANK = rank


def set_sequence_data_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    _SEQUENCE_DATA_PARALLEL_RANK = rank


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_data_parallel_rank():
    """Return my rank for the sequence data parallel group."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    if _SEQUENCE_DATA_PARALLEL_RANK is not None:
        return _SEQUENCE_DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_sequence_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_parallel_groups():
    """Set the groups to none."""
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None
    global _SEQUENCE_DATA_PARALLEL_GROUP
    _SEQUENCE_DATA_PARALLEL_GROUP = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    
def get_rank(group):
    return dist.get_rank(group=group)

def broadcast(tensor, src, group):
    torch.distributed.broadcast(tensor, src=src, group=group)

def broadcast_tensors(
    tensors: Optional[List[torch.Tensor]],
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Broadcasts a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    # share metadata first to simplify transfer
    is_src_rank = get_rank(group) == src_rank
    if is_src_rank:
        # We only want to communicate device type ie (cpu vs cuda) and not the index of cuda.
        metadata = [
            {"size": t.size(), "dtype": t.dtype, "device": torch.device(t.device.type)}
            for t in tensors
        ]
        metadata = _broadcast_object_slow(metadata, src_rank, group, dist_device)
    else:
        metadata = _broadcast_object_slow(None, src_rank, group, dist_device)

    out_tensors = []
    for i, meta in enumerate(metadata):
        if is_src_rank:
            tensor = tensors[i]
            broadcast(tensors[i].to(dist_device), src=src_rank, group=group)
        else:
            tensor = torch.zeros(
                [meta["size"].numel()], dtype=meta["dtype"], device=dist_device
            )
            broadcast(tensor, src=src_rank, group=group)

        tensor = tensor.view(meta["size"]).to(meta["device"])
        out_tensors.append(tensor)
    return out_tensors


def broadcast_object(
    obj: Any,
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    if get_rank(group) == src_rank:
        # split the tensors from the non-tensors so we can broadcast them
        # directly, avoiding unnecessary serialization/deserialization
        tensors = []
        obj = _split_tensors_from_obj(obj, tensors)
        obj = _broadcast_object_slow(obj, src_rank, group, dist_device)
        tensors = broadcast_tensors(tensors, src_rank, group, dist_device)
    else:
        obj = _broadcast_object_slow(None, src_rank, group, dist_device)
        tensors = broadcast_tensors(None, src_rank, group, dist_device)
    return _put_tensors_in_obj(obj, tensors)


def _broadcast_object_slow(
    obj: Any,
    src_rank: int,
    group: object,
    dist_device: torch.device,
) -> Any:
    if get_rank(group) == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(dist_device)
        length = torch.LongTensor([len(buffer)]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        broadcast(buffer, src=src_rank, group=group)
    else:
        # Fetch from the source
        length = torch.LongTensor([0]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        buffer = torch.ByteTensor(int(length.item())).to(dist_device)
        broadcast(buffer, src=src_rank, group=group)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location="cpu")
    return obj


@dataclass(frozen=True)
class _TensorPlaceholder:
    index: int


def _split_tensors_from_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if torch.is_tensor(obj):
        placeholder = _TensorPlaceholder(index=len(tensors))
        tensors.append(obj)
        return placeholder
    elif isinstance(obj, dict):
        return {k: _split_tensors_from_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_split_tensors_from_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_split_tensors_from_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_split_tensors_from_obj(v, tensors) for v in obj}
    else:
        return obj


def _put_tensors_in_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(obj, _TensorPlaceholder):
        return tensors[obj.index]
    elif isinstance(obj, dict):
        return {k: _put_tensors_in_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_put_tensors_in_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_put_tensors_in_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_put_tensors_in_obj(v, tensors) for v in obj}
    else:
        return obj

def split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_sequence_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # broadcast input from rank 0 to all other ranks
    rank = get_sequence_parallel_rank()
    src_rank = get_sequence_parallel_src_rank()
    group = get_sequence_parallel_group()
    broadcast(input_, src_rank, group=group)

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by sequence parallel size"
    local_dim_size = dim_size // world_size
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def split_along_second_dim(input_):
    """Split the tensor along its second dimension and keep the
    corresponding slice."""

    world_size = get_sequence_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    rank = get_sequence_parallel_rank()
    # Split along second dimension.
    dim_size = input_.size()[1]
    assert (
        dim_size % world_size == 0
    ), "Second dimension of the tensor should be divisible by sequence parallel size"
    local_dim_size = dim_size // world_size
    dim_offset = rank * local_dim_size

    output = input_[:, dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def gather_along_first_dim(input_, async_op=False, cached_buffer_name=None):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_sequence_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    if cached_buffer_name is None:
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
        )
    else:
        output = get_global_memory_buffer().get_tensor(
            dim_size, input_.dtype, cached_buffer_name
        )
    handle = torch.distributed._all_gather_base(
        output,
        input_.contiguous(),
        group=get_sequence_parallel_group(),
        async_op=async_op,
    )

    if async_op:
        # Note: [Naman] I am still not sure if this is needed but original code
        # for sequence_parallel had it, so for now keeping it.
        # Delay the start of weight gradient computation shortly (3us) to have
        # reduce scatter scheduled first and have GPU resources allocated
        _ = torch.empty(1, device=input_.device) + 1
        return output, handle

    return output


class ScatterToSequenceParallelGroup(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_first_dim(grad_output)


def scatter_to_sequence_parallel_group(input_):
    return ScatterToSequenceParallelGroup.apply(input_)
