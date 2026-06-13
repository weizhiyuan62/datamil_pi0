from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any
from urllib import error
from urllib import request

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a remote datamil-pi0 policy server in LIBERO simulation.")
    parser.add_argument("--policy-server-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--task-suite-name",
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task-ids", nargs="*", type=int, default=None, help="Subset of task ids to evaluate.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--video-out-path", default="data/libero/videos")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-path", default=None, help="Optional JSON summary path.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--http-timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
        help="Print sim-side episode progress every N executed action steps; set <=0 to disable step logs.",
    )
    parser.add_argument(
        "--state-format",
        choices=["datamil", "openpi"],
        default="datamil",
        help="datamil uses ee_pos + axisangle + [0] + first gripper qpos; openpi uses ee_pos + axisangle + full gripper qpos.",
    )
    return parser.parse_args()


class RemotePolicyClient:
    def __init__(self, base_url: str, *, timeout_sec: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.opener = request.build_opener(request.ProxyHandler({}))

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def infer(self, element: dict[str, Any]) -> dict[str, np.ndarray]:
        payload = {
            "observation/image": np.asarray(element["observation/image"], dtype=np.uint8).tolist(),
            "observation/wrist_image": np.asarray(element["observation/wrist_image"], dtype=np.uint8).tolist(),
            "observation/state": np.asarray(element["observation/state"], dtype=np.float32).tolist(),
            "prompt": str(element["prompt"]),
        }
        response = self._post_json("/act", payload)
        return {"actions": np.asarray(response["actions"], dtype=np.float32)}

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            with self.opener.open(self.base_url + path, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET {self.base_url + path} failed with HTTP {exc.code}: {body}") from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {self.base_url + path} failed with HTTP {exc.code}: {body}") from exc


def eval_libero(args: argparse.Namespace) -> None:
    import torch
    import tqdm
    from libero.libero import benchmark

    def sim_log(message: str) -> None:
        tqdm.tqdm.write(message)

    patch_torch_load_for_libero_init_states(torch)
    np.random.seed(args.seed)

    policy = RemotePolicyClient(args.policy_server_url, timeout_sec=args.http_timeout_sec)
    policy_info = policy.health()
    logging.info("Connected to policy server: %s", json.dumps(policy_info, indent=2))

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
        episode_logs = []
        try:
            for episode_idx in tqdm.tqdm(range(trial_count), desc=f"task {task_id}", leave=False):
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
                replay_images = []
                done = False
                error_message = None
                action_steps = 0
                policy_queries = 0
                t = 0
                sim_log(
                    f"[sim] start task={task_id} episode={episode_idx + 1}/{trial_count} "
                    f"max_action_steps={max_steps} desc={task_description}"
                )
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
                            policy_queries += 1
                            if len(action_chunk) < args.replan_steps:
                                raise ValueError(
                                    f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}."
                                )
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()
                        obs, _, done, _ = env.step(action.tolist())
                        action_steps += 1
                        t += 1
                        if args.log_every_steps > 0 and action_steps % args.log_every_steps == 0:
                            sim_log(
                                f"[sim] progress task={task_id} episode={episode_idx + 1}/{trial_count} "
                                f"action_steps={action_steps}/{max_steps} total_env_steps={t} "
                                f"policy_queries={policy_queries} queue={len(action_plan)}"
                            )
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                    except Exception as exc:
                        error_message = repr(exc)
                        logging.exception("Evaluation exception on task_id=%s episode=%s", task_id, episode_idx)
                        break

                task_episodes += 1
                total_episodes += 1
                status = "success" if done else "error" if error_message is not None else "failure"
                sim_log(
                    f"[sim] end task={task_id} episode={episode_idx + 1}/{trial_count} "
                    f"status={status} action_steps={action_steps} total_env_steps={t} "
                    f"policy_queries={policy_queries} success_rate_task={task_successes}/{task_episodes}"
                )
                episode_logs.append(
                    {
                        "episode_index": int(episode_idx),
                        "status": status,
                        "success": bool(done),
                        "action_steps": int(action_steps),
                        "total_env_steps": int(t),
                        "policy_queries": int(policy_queries),
                        "error": error_message,
                    }
                )
                if not args.no_video and replay_images:
                    import imageio

                    suffix = "success" if done else "failure"
                    task_segment = str(task_description).replace(" ", "_").replace("/", "_")[:120]
                    imageio.mimwrite(
                        video_out_path / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
        finally:
            close_env(env)

        task_success_rate = float(task_successes) / max(1, task_episodes)
        total_success_rate = float(total_successes) / max(1, total_episodes)
        results.append(
            {
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": task_success_rate,
                "episode_logs": episode_logs,
            }
        )
        logging.info("Task %s success rate: %.3f", task_id, task_success_rate)
        logging.info("Running total success rate: %.3f", total_success_rate)

    summary = {
        "policy_server_url": args.policy_server_url,
        "policy_server_info": policy_info,
        "task_suite_name": args.task_suite_name,
        "task_ids": [int(x) for x in task_ids],
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": float(total_successes) / max(1, total_episodes),
        "results": results,
    }
    summary_path = output_path(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Wrote eval summary to %s", summary_path)
    logging.info("Total success rate: %.3f", summary["total_success_rate"])


def output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return Path(args.output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/libero/eval_results") / f"libero_eval_{args.task_suite_name}_{timestamp}.json"


def patch_torch_load_for_libero_init_states(torch_module) -> None:
    if getattr(torch_module.load, "_datamil_pi0_libero_compat", False):
        return
    original_load = torch_module.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_default._datamil_pi0_libero_compat = True
    torch_module.load = load_with_legacy_default


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


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    new_width = max(1, int(round(image_width * scale)))
    new_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


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
