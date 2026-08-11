# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
import torch.nn.functional as nnf
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = sorted(v.split('/meta/info.json')[0] for v in repo_list)
    if num_init_worker <= 1 or len(repo_list) <= 1:
        return [construct_func(repo_id) for repo_id in repo_list]

    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        if hasattr(config, 'dataset_init_worker'):
            num_init_worker = config.dataset_init_worker
        elif hasattr(config, 'load_worker'):
            num_init_worker = min(num_init_worker, max(1, int(config.load_worker)))
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           max(1, int(num_init_worker)),
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.config = config
        self.action_key = getattr(config, 'action_key', 'action')
        self.load_actions_direct_from_parquet = bool(
            getattr(config, 'load_actions_direct_from_parquet', False)
        )
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        if self.load_actions_direct_from_parquet:
            missing_data_files = [
                self.root / self.meta.get_data_file_path(episode_index)
                for episode_index in sorted(self.meta.episodes)
                if not (
                    self.root / self.meta.get_data_file_path(episode_index)
                ).is_file()
            ]
            if missing_data_files:
                raise FileNotFoundError(
                    f"Missing {len(missing_data_files)} episode Parquet files; "
                    f"first missing file: {missing_data_files[0]}"
                )
            self.hf_dataset = None
            self._direct_action_cache = {}
        else:
            try:
                assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
                self.hf_dataset = self.load_hf_dataset()
            except (AssertionError, FileNotFoundError, NotADirectoryError):
                self.revision = get_safe_version(self.repo_id, self.revision)
                self.download_episodes(download_videos)
                self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = None if self.hf_dataset is None else self.hf_dataset.with_format(
                type='torch',
                columns=[self.action_key],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _load_episode_actions(self, episode_index):
        actions = self._direct_action_cache.get(episode_index)
        if actions is not None:
            return actions

        import pyarrow.parquet as pq

        parquet_path = self.root / self.meta.get_data_file_path(episode_index)
        try:
            table = pq.read_table(parquet_path, columns=[self.action_key])
            actions = np.asarray(
                table[self.action_key].to_pylist(),
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read action column {self.action_key!r} "
                f"from {parquet_path}"
            ) from exc

        expected_length = int(self.meta.episodes[episode_index]["length"])
        expected_dim = int(getattr(self.config, "action_source_dim", actions.shape[-1]))
        if actions.ndim != 2:
            raise ValueError(
                f"Expected 2-D actions in {parquet_path}, got shape {actions.shape}"
            )
        if actions.shape != (expected_length, expected_dim):
            raise ValueError(
                f"Expected actions with shape ({expected_length}, {expected_dim}) "
                f"in {parquet_path}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"Non-finite actions found in {parquet_path}")

        actions = np.ascontiguousarray(actions)
        self._direct_action_cache[episode_index] = actions
        return actions

    def _get_range_hf_data(
        self,
        start_frame,
        end_frame,
        episode_index=None,
    ):
        if self._hf_torch_view is None:
            if episode_index is None:
                raise ValueError("episode_index is required for direct action loading")
            actions = self._load_episode_actions(episode_index)
            if not 0 <= start_frame < end_frame <= len(actions):
                raise IndexError(
                    f"Invalid action range [{start_frame}:{end_frame}] for "
                    f"episode {episode_index} with {len(actions)} frames"
                )
            return {
                self.action_key: torch.from_numpy(actions[start_frame:end_frame])
            }
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        prefix_action_num = frame_stride * 4
        action = np.pad(action, pad_width=((prefix_action_num, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        if getattr(self.config, "initial_action_condition", None) == "model_zero":
            # F0 is a non-executed BOS condition. Mask it before normalization
            # so asymmetric physical statistics cannot turn it into an extreme.
            action_mask[:prefix_action_num] = False
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        if getattr(self.config, "initial_action_condition", None) == "model_zero":
            if np.any(action_aligned[:, 0]) or np.any(action_mask_aligned[:, 0]):
                raise AssertionError(
                    "Initial action condition must be normalized zero and masked"
                )
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        if self._hf_torch_view is None:
            hf_data_frames = self._get_range_hf_data(
                local_start_frame,
                local_end_frame,
                episode_index=episode_index,
            )
        else:
            start_frame = self._get_global_idx(episode_index, start_frame)
            end_frame = self._get_global_idx(episode_index, end_frame)
            hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict[self.action_key])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)


def pad_collate_fn(batch):
    """Pad variable-length latents/actions to the batch's max frame count.

    Each sample's 'latents' (C,F,H,W) and 'actions'/'actions_mask' (30,F,N,1)
    have a variable F (RoboMME episode/segment length). Zero-pad every sample
    to F_max on the frame axis and emit 'frame_valid_mask' (B,F_max) bool
    (True=real frame, False=padding) so the loss and attention mask can
    exclude padded frames. actions_mask is padded with False, which the
    existing action-loss masking in train.py's compute_loss already respects
    unchanged.
    """
    f_max = max(item['latents'].shape[1] for item in batch)
    latents, actions, actions_masks, text_embs, frame_valid_masks = [], [], [], [], []
    for item in batch:
        f = item['latents'].shape[1]
        pad_f = f_max - f
        latents.append(nnf.pad(item['latents'], (0, 0, 0, 0, 0, pad_f)))
        actions.append(nnf.pad(item['actions'], (0, 0, 0, 0, 0, pad_f)))
        actions_masks.append(nnf.pad(item['actions_mask'], (0, 0, 0, 0, 0, pad_f), value=False))
        text_embs.append(item['text_emb'])
        mask = torch.zeros(f_max, dtype=torch.bool)
        mask[:f] = True
        frame_valid_masks.append(mask)
    return {
        'latents': torch.stack(latents),
        'actions': torch.stack(actions),
        'actions_mask': torch.stack(actions_masks),
        'text_emb': torch.stack(text_embs),
        'frame_valid_mask': torch.stack(frame_valid_masks),
    }


if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
