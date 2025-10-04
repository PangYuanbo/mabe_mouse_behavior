"""
Mouse Pair ID Mapping for V9
Maps (agent_id, target_id) to directional pair IDs (0-11)
"""

import numpy as np


def get_pair_id(agent_id: int, target_id: int) -> int:
    """
    Map (agent_id, target_id) to a unique pair ID (0-11)

    4 mice × 3 possible targets (excluding self) = 12 directed pairs

    Args:
        agent_id: Agent mouse ID (0-3)
        target_id: Target mouse ID (0-3)

    Returns:
        pair_id: Unique ID (0-11)

    Example:
        (0, 1) -> 0
        (0, 2) -> 1
        (0, 3) -> 2
        (1, 0) -> 3
        (1, 2) -> 4
        (1, 3) -> 5
        ...
    """
    if agent_id == target_id:
        raise ValueError(f"Agent and target cannot be the same: {agent_id}")

    if not (0 <= agent_id < 4 and 0 <= target_id < 4):
        raise ValueError(f"Invalid agent/target IDs: {agent_id}, {target_id}")

    # Adjust target index to skip diagonal
    if target_id > agent_id:
        adjusted_target = target_id - 1
    else:
        adjusted_target = target_id

    pair_id = agent_id * 3 + adjusted_target
    return pair_id


def get_agent_target_from_pair_id(pair_id: int) -> tuple:
    """
    Inverse mapping: recover (agent_id, target_id) from pair_id

    Args:
        pair_id: Unique pair ID (0-11)

    Returns:
        (agent_id, target_id): Tuple of mouse IDs
    """
    if not (0 <= pair_id < 12):
        raise ValueError(f"Invalid pair_id: {pair_id}")

    agent_id = pair_id // 3
    target_offset = pair_id % 3

    # Restore original target_id
    if target_offset >= agent_id:
        target_id = target_offset + 1
    else:
        target_id = target_offset

    return agent_id, target_id


def get_channel_index(action_id: int, agent_id: int, target_id: int, num_actions: int = 28) -> int:
    """
    Get the channel index for (action, agent, target) triplet

    Total channels = num_actions × 12 pairs

    Args:
        action_id: Action class ID (0-27)
        agent_id: Agent mouse ID (0-3)
        target_id: Target mouse ID (0-3)
        num_actions: Number of action classes (default 28)

    Returns:
        channel_idx: Channel index in boundary heatmap
    """
    pair_id = get_pair_id(agent_id, target_id)
    channel_idx = action_id * 12 + pair_id
    return channel_idx


def get_action_pair_from_channel(channel_idx: int, num_actions: int = 28) -> tuple:
    """
    Inverse mapping: recover (action_id, agent_id, target_id) from channel index

    Args:
        channel_idx: Channel index in boundary heatmap
        num_actions: Number of action classes (default 28)

    Returns:
        (action_id, agent_id, target_id): Tuple of IDs
    """
    action_id = channel_idx // 12
    pair_id = channel_idx % 12
    agent_id, target_id = get_agent_target_from_pair_id(pair_id)
    return action_id, agent_id, target_id


# Unit tests
if __name__ == '__main__':
    print("Testing Mouse Pair Mapping...")

    # Test all 12 pairs
    print("\n[Test 1] All 12 directed pairs:")
    for agent in range(4):
        for target in range(4):
            if agent == target:
                continue
            pair_id = get_pair_id(agent, target)
            recovered_agent, recovered_target = get_agent_target_from_pair_id(pair_id)
            print(f"  ({agent}, {target}) -> pair_id={pair_id:2d} -> ({recovered_agent}, {recovered_target})")
            assert agent == recovered_agent and target == recovered_target

    # Test channel indexing
    print("\n[Test 2] Channel indexing:")
    for action_id in [0, 5, 15, 27]:
        for agent_id in range(4):
            for target_id in range(4):
                if agent_id == target_id:
                    continue
                channel = get_channel_index(action_id, agent_id, target_id)
                rec_action, rec_agent, rec_target = get_action_pair_from_channel(channel)
                print(f"  action={action_id:2d}, agent={agent_id}, target={target_id} -> channel={channel:3d} -> ({rec_action:2d}, {rec_agent}, {rec_target})")
                assert action_id == rec_action and agent_id == rec_agent and target_id == rec_target

    # Test edge cases
    print("\n[Test 3] Edge cases:")
    try:
        get_pair_id(2, 2)  # Same agent and target
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Caught expected error: {e}")

    try:
        get_pair_id(5, 2)  # Invalid agent
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Caught expected error: {e}")

    print("\n[OK] All tests passed!")
