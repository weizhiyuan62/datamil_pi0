from __future__ import annotations

import argparse
import collections
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained datamil-pi0 checkpoint in LIBERO simulation.")
    parser.add_argument("--checkpoint-dir", required=True, help="Step checkpoint directory containing model.safetensors.")
    parser.add_argument("--norm-stats-path", default=None, help="Optional norm_stats.json override.")
    parser.add_argument("--task-suite-name", default="libero_10", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-ids", nargs="*", type=int, default=None, help="Subset of task ids to evaluate.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--video-out-path", default="data/libero/videos")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None, help="Override model dtype from metadata.")
    parser.add_argument(
        "--state-format",
        choices=["datamil", "openpi"],
        default="datamil",
        help="datamil uses ee_pos + axisangle + [0] + first gripper qpos; openpi uses ee_pos + axisangle + full gripper qpos.",
    )
    parser.add_argument(
        "--gripper-conversion",
        choices=["datamil", "none"],
        default="datamil",
        help="datamil maps predicted gripper g in [0,1] back to env action 1-2*g.",
    )
    return parser.parse_args()


def load_model_config(checkpoint_dir: Path, *, dtype_override: str | None) -> Pi0Config:
    from datamil_pi0.model.config import Pi0Config

    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model_payload = metadata.get("config", {}).get("model", {})
        valid_fields = set(Pi0Config.__dataclass_fields__)
        kwargs = {key: value for key, value in model_payload.items() if key in valid_fields}
        if dtype_override is not None:
            kwargs["dtype"] = dtype_override
        return Pi0Config(**kwargs)
    config = Pi0Config()
    if dtype_override is not None:
        config = Pi0Config(dtype=dtype_override)
    return config


def find_norm_stats_path(checkpoint_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    info_path = checkpoint_dir / "norm_stats_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        rel_path = info.get("checkpoint_relative_path")
        if rel_path is not None and (checkpoint_dir / rel_path).exists():
            return checkpoint_dir / rel_path
    candidates = sorted((checkpoint_dir / "assets").glob("*/norm_stats.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No norm_stats.json found under {checkpoint_dir / 'assets'}")
    raise ValueError(f"Multiple norm_stats.json files found; pass --norm-stats-path explicitly: {candidates}")


def make_inference_transform(config: Pi0Config, norm_stats):
    from datamil_pi0.tokenizer import PaligemmaTokenizer
    from datamil_pi0.transforms import Compose
    from datamil_pi0.transforms import LiberoInputs
    from datamil_pi0.transforms import Normalize
    from datamil_pi0.transforms import PadStatesAndActions
    from datamil_pi0.transforms import RepackTransform
    from datamil_pi0.transforms import ResizeImages
    from datamil_pi0.transforms import TokenizePrompt

    return Compose(
        [
            RepackTransform(
                {
                    "observation/image": ("observation/image", "image"),
                    "observation/wrist_image": ("observation/wrist_image", "wrist_image"),
                    "observation/state": ("observation/state", "state"),
                    "prompt": "prompt",
                }
            ),
            LiberoInputs(),
            Normalize(norm_stats, use_quantiles=False),
            ResizeImages(224, 224),
            TokenizePrompt(PaligemmaTokenizer(config.max_token_len), discrete_state_input=config.discrete_state_input),
            PadStatesAndActions(config.action_dim),
        ]
    )


def collate_one(item: dict[str, Any]) -> dict[str, Any]:
    from datamil_pi0.utils import tree_map

    def add_batch(x):
        return np.expand_dims(np.asarray(x), axis=0)

    return tree_map(add_batch, item)


class LocalPi0LiberoPolicy:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        norm_stats_path: Path,
        device: torch.device,
        dtype_override: str | None,
        gripper_conversion: str,
        num_inference_steps: int,
    ):
        import safetensors.torch
        from datamil_pi0.model.pi0 import PI0Pytorch
        from datamil_pi0.transforms import load_norm_stats

        self.device = device
        self.config = load_model_config(checkpoint_dir, dtype_override=dtype_override)
        self.norm_stats = load_norm_stats(norm_stats_path)
        self.transform = make_inference_transform(self.config, self.norm_stats)
        self.model = PI0Pytorch(self.config).to(device)
        safetensors.torch.load_model(self.model, checkpoint_dir / "model.safetensors")
        self.model.eval()
        self.gripper_conversion = gripper_conversion
        self.num_inference_steps = int(num_inference_steps)

    def infer(self, element: dict[str, Any]) -> dict[str, np.ndarray]:
        import torch
        from datamil_pi0.model.observation import Observation
        from datamil_pi0.utils import tree_map
        from datamil_pi0.utils import tree_to_device

        item = self.transform(element)
        batch = tree_map(torch.as_tensor, collate_one(item))
        observation = tree_to_device(Observation.from_dict(batch), self.device)
        with torch.no_grad():
            actions = self.model.sample_actions(self.device, observation, num_steps=self.num_inference_steps)
        actions = actions.detach().cpu().numpy()[0]
        actions = unnormalize_actions(actions, self.norm_stats)[:, :7]
        if self.gripper_conversion == "datamil":
            actions[:, -1] = 1.0 - 2.0 * actions[:, -1]
        return {"actions": actions.astype(np.float32)}


def unnormalize_actions(actions: np.ndarray, norm_stats) -> np.ndarray:
    stats = norm_stats["actions"]
    mean = pad_to_dim(stats.mean, actions.shape[-1], value=0.0)
    std = pad_to_dim(stats.std, actions.shape[-1], value=1.0)
    return actions * (std + 1e-6) + mean


def pad_to_dim(x: np.ndarray, target_dim: int, *, value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - x.shape[-1])
    return np.pad(x, pad_width, constant_values=value)


def eval_libero(args: argparse.Namespace) -> None:
    import torch
    import tqdm
    from datamil_pi0.transforms import resize_with_pad
    from libero.libero import benchmark

    np.random.seed(args.seed)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    norm_stats_path = find_norm_stats_path(checkpoint_dir, args.norm_stats_path)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logging.info("Loading checkpoint from %s", checkpoint_dir)
    logging.info("Using norm stats from %s", norm_stats_path)
    policy = LocalPi0LiberoPolicy(
        checkpoint_dir,
        norm_stats_path=norm_stats_path,
        device=device,
        dtype_override=args.dtype,
        gripper_conversion=args.gripper_conversion,
        num_inference_steps=args.num_inference_steps,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_ids = args.task_ids if args.task_ids is not None and len(args.task_ids) > 0 else list(range(task_suite.n_tasks))
    max_steps = max_steps_for_suite(args.task_suite_name)

    video_out_path = Path(args.video_out_path)
    if not args.no_video:
        video_out_path.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    results = []
    for task_id in tqdm.tqdm(task_ids, desc="libero tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0
        trial_count = min(args.num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(trial_count), desc=f"task {task_id}", leave=False):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            replay_images = []
            done = False
            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = resize_with_pad(img, args.resize_size, args.resize_size).astype(np.uint8)
                    wrist_img = resize_with_pad(wrist_img, args.resize_size, args.resize_size).astype(np.uint8)
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": libero_state(obs, state_format=args.state_format),
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(element)["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}."
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception:
                    logging.exception("Evaluation exception on task_id=%s episode=%s", task_id, episode_idx)
                    break

            task_episodes += 1
            total_episodes += 1
            if not args.no_video and replay_images:
                import imageio

                suffix = "success" if done else "failure"
                task_segment = str(task_description).replace(" ", "_").replace("/", "_")[:120]
                imageio.mimwrite(
                    video_out_path / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

        task_success_rate = float(task_successes) / max(1, task_episodes)
        total_success_rate = float(total_successes) / max(1, total_episodes)
        results.append(
            {
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": task_success_rate,
            }
        )
        logging.info("Task %s success rate: %.3f", task_id, task_success_rate)
        logging.info("Running total success rate: %.3f", total_success_rate)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_path": str(norm_stats_path),
        "task_suite_name": args.task_suite_name,
        "task_ids": [int(x) for x in task_ids],
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": float(total_successes) / max(1, total_episodes),
        "results": results,
    }
    summary_path = checkpoint_dir / f"libero_eval_{args.task_suite_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Wrote eval summary to %s", summary_path)
    logging.info("Total success rate: %.3f", summary["total_success_rate"])


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def libero_state(obs: dict[str, Any], *, state_format: str) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_axisangle = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)).astype(np.float32)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    if state_format == "openpi":
        return np.concatenate([eef_pos, eef_axisangle, gripper_qpos], axis=-1).astype(np.float32)
    return np.concatenate([eef_pos, eef_axisangle, np.zeros((1,), dtype=np.float32), gripper_qpos[:1]], axis=-1).astype(np.float32)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    eval_libero(parse_args())


if __name__ == "__main__":
    main()
