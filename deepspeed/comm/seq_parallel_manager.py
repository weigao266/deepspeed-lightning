"""Lightning sequence parallel groups."""

import torch
from .seq_parallel_utils import GlobalMemoryBuffer

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None

# For DeepSpeed's sequence parallel
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


def initialize_seq_parallel(
    data_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
) -> None:
    """Initialize parallel groups."""
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    enable_ds_sequence_parallel = sequence_parallel_size > 1
    if enable_ds_sequence_parallel:
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
    global _DATA_PARALLEL_GROUP_GLOO
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
    if enable_ds_sequence_parallel:
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
    

def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_sequence_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_sequence_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_first_dim(input_, async_op=False, cached_buffer_name=None):
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


class _ScatterToSequenceParallelGroup(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


def scatter_to_sequence_parallel_group(input_):
    return _ScatterToSequenceParallelGroup.apply(input_)
