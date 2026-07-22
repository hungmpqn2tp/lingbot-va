from types import SimpleNamespace

import numpy as np

from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset


def test_robomme_initial_action_is_model_zero_and_not_a_target():
    dataset = object.__new__(LatentLeRobotDataset)
    used_channels = list(range(7)) + [28]
    inverse_channels = [8] * 30
    for source_index, model_channel in enumerate(used_channels):
        inverse_channels[model_channel] = source_index

    dataset.config = SimpleNamespace(
        env_type="none",
        inverse_used_action_channel_ids=inverse_channels,
        initial_action_condition="model_zero",
    )
    # Deliberately asymmetric: raw physical zero would normalize to -1.
    dataset.q01 = np.zeros((1, 30), dtype=np.float32)
    dataset.q99 = np.full((1, 30), 2.0, dtype=np.float32)

    action, action_mask = dataset._action_post_process(
        local_start_frame=0,
        local_end_frame=4,
        latent_frame_ids=np.arange(5),
        action=np.ones((4, 8), dtype=np.float32),
    )

    assert tuple(action.shape) == (30, 2, 4, 1)
    assert tuple(action_mask.shape) == (30, 2, 4, 1)
    assert not action[:, 0].any()
    assert not action_mask[:, 0].any()
    assert action_mask[used_channels, 1].all()
    unused_channels = sorted(set(range(30)) - set(used_channels))
    assert not action_mask[unused_channels, 1].any()
