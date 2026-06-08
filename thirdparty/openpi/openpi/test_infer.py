import os
import sys
import json

import argparse
import torch
import safetensors
from safetensors.torch import load_file, save_file
import numpy as np
import pickle

from transformers.cache_utils import DynamicCache
from openpi.policies import policy_config
from openpi.policies import droid_policy
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from openpi.models_pytorch import pi0_pytorch
# config = _config.get_config("pi0_droid")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference of a trained policy.")
    parser.add_argument("--mode", choices=["reset", "forward", "denoise", "default"], default="default", help="Whether to include the third_party_safetensors directory in the path.")
    args = parser.parse_args()

    example_path = "tmp/example.pkl"
    if not os.path.exists(example_path):
        example = droid_policy.make_droid_example()
        with open(example_path, "wb") as f:
            pickle.dump(example, f)

    example = pickle.load(open(example_path, "rb"))
    print("Example loaded, starting inference...")
    config = _config.get_config("pi0_droid")
    if args.mode == "reset":
        weight_path = "tmp/full_converted_vla_model.safetensors"
        pi0config = config
        model = pi0_pytorch.PI0Pytorch(config=pi0config.model)
        safetensors.torch.load_model(model, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    elif args.mode == "forward":
        print("Loading few policy...")
        # few means we load some weights from outside
        checkpoint_dir = ".cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch"
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        pi0config = config
        # create default model, with pi0's weights
        model = pi0_pytorch.PI0Pytorch(config=pi0config.model)
        safetensors.torch.load_model(model, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # load our vla converted weights, and overwrite the corresponding weights in the default model
        converted_weight_path = "tmp/converted_vla_model.safetensors"
        model.load_state_dict(load_file(converted_weight_path), strict=False)

        paligemma_input = pickle.load(open("tmp/paligemma_input.pkl", "rb"))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        target_dtype = torch.bfloat16 
        model.to(device)
        model.eval()

        attention_mask=paligemma_input["attention_mask"].to(device)
        position_ids=paligemma_input["position_ids"].to(device)
        past_key_values=paligemma_input["past_key_values"]
        inputs_embeds=paligemma_input["inputs_embeds"]
        inputs_embeds[0].to(device)
        use_cache=paligemma_input["use_cache"]
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=target_dtype):
                result = model.paligemma_with_expert.forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache
                )
                print(result)

    elif args.mode == "denoise":
        print("Loading few policy...")
        # few means we load some weights from outside
        pi0config = config
        # create default model, with pi0's weights
        converted_weight_path = "tmp/converted_vla_model.safetensors"
        model = pi0_pytorch.PI0Pytorch(config=pi0config.model)
        model.load_state_dict(load_file(converted_weight_path), strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # load our vla converted weights, and overwrite the corresponding weights in the default model
        
        

        paligemma_input = pickle.load(open("tmp/paligemma_input.pkl", "rb"))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        target_dtype = torch.bfloat16 
        model.to(device)
        model.eval()

        def generate_flow_matching(
            prefix_pad_masks: torch.FloatTensor,    # update_valid_mask in our code
            
            proprio: torch.FloatTensor,
            denoise_step: int = 10, 
            update_block_mask: torch.FloatTensor | None = None,
            noise: torch.FloatTensor | None = None, 
            past_key_values: DynamicCache | None = None, 
            proprio_emb: torch.FloatTensor | None = None, 
        ):
            #input: time, x_t(noise), num_steps, past_key_values, prefix_pad_masks(img, text)
            # state : [B, T, D_action]
            # pi0_take_state: [T, D_action]
            dt = -1.0 / denoise_step
            dt = torch.tensor(dt, dtype=torch.float32, device=device)
            state = proprio[0]
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)

            bsize = 1 # our test case only has batch size of 1
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = model.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )

                # Euler step - use new tensor assignment instead of in-place operation
                x_t = x_t + dt * v_t
                time += dt
            return x_t
        with open("test_input/10-57-30/generate_flow_matching_input.pkl", "rb") as f:
            generate_flow_matching_input = pickle.load(f)
            update_block_mask=generate_flow_matching_input["update_block_mask"].to(device)
            prefix_pad_masks=generate_flow_matching_input["update_valid_mask"].to(device)
            proprio=generate_flow_matching_input["proprio"].to(device)
            denoise_step=generate_flow_matching_input["denoise_step"]
            past_key_values=generate_flow_matching_input["past_key_values"]
        noise_path = "test_input/10-57-30/noise.pt"
        noise = torch.load(noise_path).to(device)

        output = generate_flow_matching(
            prefix_pad_masks=prefix_pad_masks,
            proprio=proprio,
            denoise_step=denoise_step,
            noise=noise,
            past_key_values=past_key_values,
        )
        with open("test_input/10-57-30/generate_flow_matching_output.pkl", "wb") as f:
            std_output = pickle.load(f)
        print("Generated output:", output)
        print("Standard output:", std_output)


    elif args.mode == "default":
        checkpoint_dir = ".cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch"
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        result = policy.infer(example)["actions"]

    print(f"finish all")

    