from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_pi0_libero import LocalPi0LiberoPolicy  # noqa: E402
from eval_pi0_libero import find_norm_stats_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a datamil-pi0 checkpoint as a small HTTP LIBERO policy.")
    parser.add_argument("--checkpoint-dir", required=True, help="Step checkpoint directory containing model.safetensors.")
    parser.add_argument("--norm-stats-path", default=None, help="Optional norm_stats.json override.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None, help="Override model dtype from metadata.")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--gripper-conversion",
        choices=["datamil", "none"],
        default="datamil",
        help="datamil maps predicted gripper g in [0,1] back to LIBERO env action 1-2*g.",
    )
    parser.add_argument("--max-request-mb", type=float, default=32.0)
    parser.add_argument("--log-prompts", type=int, default=5, help="Log the first N prompts received by /act.")
    return parser.parse_args()


class PolicyServerState:
    policy: LocalPi0LiberoPolicy | None = None
    info: dict[str, Any] = {}
    max_request_bytes: int = 32 * 1024 * 1024
    log_prompts_remaining: int = 0
    request_count: int = 0


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server_version = "datamil-pi0-policy/0.1"

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._write_json({"ok": True, **PolicyServerState.info})
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/act":
            self.send_error(404, "unknown endpoint")
            return
        if PolicyServerState.policy is None:
            self.send_error(503, "policy is not loaded")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        if content_length <= 0:
            self.send_error(400, "empty request body")
            return
        if content_length > PolicyServerState.max_request_bytes:
            self.send_error(413, "request body too large")
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            element = {
                "observation/image": np.asarray(payload["observation/image"], dtype=np.uint8),
                "observation/wrist_image": np.asarray(payload["observation/wrist_image"], dtype=np.uint8),
                "observation/state": np.asarray(payload["observation/state"], dtype=np.float32),
                "prompt": str(payload["prompt"]),
            }
            result = PolicyServerState.policy.infer(element)
            PolicyServerState.request_count += 1
            if PolicyServerState.log_prompts_remaining > 0:
                PolicyServerState.log_prompts_remaining -= 1
                logging.info(
                    "Policy request %s prompt=%r state_shape=%s actions_shape=%s",
                    PolicyServerState.request_count,
                    element["prompt"],
                    tuple(element["observation/state"].shape),
                    tuple(result["actions"].shape),
                )
            self._write_json({"actions": result["actions"].tolist()})
        except KeyError as exc:
            self.send_error(400, f"missing request key: {exc}")
        except Exception as exc:
            logging.exception("policy inference failed")
            self.send_error(500, f"policy inference failed: {exc}")

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _write_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import torch

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

    PolicyServerState.policy = policy
    PolicyServerState.max_request_bytes = int(args.max_request_mb * 1024 * 1024)
    PolicyServerState.log_prompts_remaining = max(0, int(args.log_prompts))
    PolicyServerState.info = {
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_path": str(norm_stats_path),
        "device": str(device),
        "dtype": policy.config.dtype,
        "action_dim": int(policy.config.action_dim),
        "action_horizon": int(policy.config.action_horizon),
        "num_inference_steps": int(args.num_inference_steps),
        "gripper_conversion": args.gripper_conversion,
    }

    server = ThreadingHTTPServer((args.host, args.port), PolicyRequestHandler)
    logging.info("Serving policy at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down policy server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
