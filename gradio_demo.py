import os, sys
import argparse
import socket

from pathlib import Path

import numpy as np
import torch
import gradio as gr
import traceback

from einops import rearrange
from VidUtil import Video

from src.demo.camc2v_demo import CamC2VDemo
from src.data import get_realestate10k

def get_models(path: Path) -> tuple[list[Path], dict]:
    dirs = [d for d in path.iterdir() if d.is_dir()]
    valid_dirs = []
    checkpoint_map = {}
    for d in dirs:
        config_path = d / "config.yaml"
        ckpt_path = d / "checkpoints"
        if config_path.exists() and ckpt_path.exists():
            valid_dirs.append(d)
            ckpts = {}
            # Look for actual checkpoint directories or .ckpt files
            for ckpt in ckpt_path.iterdir():
                if ckpt.is_dir():
                    # Look for the actual checkpoint file
                    pt_file = ckpt / "checkpoint" / "mp_rank_00_model_states.pt"
                    if pt_file.exists():
                        ckpts[ckpt.name] = pt_file
                elif ckpt.suffix == ".ckpt":
                    ckpts[ckpt.stem] = ckpt
            if ckpts:
                checkpoint_map[d] = ckpts
            
    return valid_dirs, checkpoint_map

def sample_batch(
    batch: dict,
    frame_stride: int,
    video_length: int,
    start_index: int = 0,
    condition_indices: list[int] = [],
    add_batch_dim: bool = True,
):
    
    C, T, H, W = batch["video"].shape

    selected_indices = list(range(start_index, start_index + video_length * frame_stride, frame_stride))
    if selected_indices[-1] >= T:
        print("Warning: selected indices exceed video length. Adjusting to fit within bounds.")
        return None
    
    batch["cond_frames"] = batch["video"][:, condition_indices, :, :]
    batch["RT_cond"] = batch["RT"][condition_indices, :, :]
    batch["camera_intrinsics_cond"] = batch["camera_intrinsics"][condition_indices, :, :]
    if "depth_maps" in batch:
        batch["depth_maps_cond"] = batch["depth_maps"][condition_indices, ...]
    if "RT_depth" in batch:
        batch["RT_depth_cond"] = batch["RT_depth"][condition_indices, :, :]
        batch["camera_intrinsics_depth_cond"] = batch["camera_intrinsics_depth"][condition_indices, :, :]

    batch["video"] = batch["video"][:, selected_indices, :, :]
    batch["RT"] = batch["RT"][selected_indices, :, :]
    batch["camera_intrinsics"] = batch["camera_intrinsics"][selected_indices, :, :]
    batch["depth_maps"] = batch["depth_maps"][selected_indices, ...]
    if "RT_depth" in batch:
        batch["RT_depth"] = batch["RT_depth"][selected_indices, :, :]
        batch["camera_intrinsics_depth"] = batch["camera_intrinsics_depth"][selected_indices,...]
    batch["frame_stride"] = frame_stride

    if add_batch_dim:
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].unsqueeze(0)  # Add batch dimension
    return batch


def gradio_app(path: Path, machine: str, output: Path):

    print("Initializing gradio demo")
    # Ensure output directory exists
    output.mkdir(parents=True, exist_ok=True)
    
    model_paths, checkpoint_map = get_models(path)
    model_map = {p.name: p for p in model_paths}

    # Create flat checkpoint map for dropdown
    ckpt_choices = {}
    for model_path, ckpts in checkpoint_map.items():
        model_name = model_path.name
        ckpt_choices[model_name] = list(ckpts.keys())

    demo = CamC2VDemo()

    dataset = get_realestate10k(machine, frame_stride=1, video_length=-1)

    video_names = [Path(p).stem for p in dataset.video_names]

    with gr.Blocks(analytics_enabled=False) as app:
        
        with gr.Row():
            video_dropdown = gr.Dropdown(label="Video Selection", elem_id="video_dropdown", choices=video_names, value=None)
        with gr.Row():
            frame_gallery = gr.Gallery(label="Frames (Indexed)", columns=5)
        with gr.Row():
            reference_selector = gr.CheckboxGroup(
                choices=[], 
                label="Select Reference frame", 
                info="Select single frame to start generation from"
            )
            frame_selector = gr.CheckboxGroup(
                choices=[], 
                label="Select Condition Frames", 
                info="Select frames by index"
            )
        
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Model", elem_id='model_dd', choices=list(model_map.keys()))
            ckpt_dropdown = gr.Dropdown(label="Checkpoint", elem_id='ckpt_dd', choices=[])
        with gr.Row():
            load_btn = gr.Button("Load Model")
        with gr.Row():
            gen_btn = gr.Button("Generate")
        with gr.Row():
            gt_video = gr.Video(label="Ground Truth Video", elem_id="gt_vid", interactive=False, autoplay=True, loop=True)
            gen_vid = gr.Video(label="Generated Video", elem_id="gen_vid", interactive=False, autoplay=True, loop=True)
        
        with gr.Row():
            with gr.Column():
                trace_extract_ratio = gr.Slider(minimum=0, maximum=1.0, step=0.1, elem_id="trace_extract_ratio", label="Trace Extract Ratio", value=0.1)
                trace_scale_factor = gr.Slider(minimum=0, maximum=5, step=0.1, elem_id="trace_scale_factor", label="Camera Trace Scale Factor", value=1.0)
                auto_reg_steps = gr.Slider(minimum=0, maximum=10, step=1, elem_id="auto_reg_steps", label="Auto-regressive Steps", value=0)

            with gr.Column():
                enable_camera_condition = gr.Checkbox(label='Enable Camera Condition', elem_id="enable_camera_condition", value=True)
                camera_cfg = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, elem_id="Camera CFG", label="Camera CFG", value=1.0, visible=False)
                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=3.5, elem_id="cfg_scale")
                frame_stride = gr.Slider(minimum=1, maximum=10, step=1, label='Frame Stride', value=2, elem_id="frame_stride")
                steps = gr.Slider(minimum=1, maximum=250, step=1, elem_id="steps", label="Sampling Steps (DDPM)", value=25)
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**31, step=1, value=12333)

        ## Callback functions
        def generate(
            model_name: str,
            reference_indices: list[str],  # Changed from reference_index
            cond_frame_indices: list[str],
            video_name: str,

            trace_extract_ratio: float,
            trace_scale_factor: float,
            auto_reg_steps: int,
            enable_camera_condition: bool,
            cfg_scale: float,
            frame_stride: int,
            steps: int,
            seed: int,

        ):
            if not model_name or not video_name:
                return None
                
            if not reference_indices or len(reference_indices) != 1:
                print("Please select exactly one reference frame")
                return None
                
            try:
                reference_index = int(reference_indices[0])
                cond_indices = [int(idx) for idx in cond_frame_indices] if cond_frame_indices else []
                
                index = dataset.get_index_by_name(video_name)
                batch = dataset[index]
                import ipdb; ipdb.set_trace()
                batch = sample_batch(
                    batch,
                    frame_stride=frame_stride,
                    video_length=16,
                    start_index=reference_index,
                    condition_indices=cond_indices
                )

                if batch is None:
                    print("Failed to create batch - check frame indices and video length")
                    return None, None

                output_path = output / model_name / video_name
                output_path.mkdir(parents=True, exist_ok=True)

                demo.generate(
                    video = rearrange(batch['video'], "B C T H W -> B T C H W"), # [B, T, C, H, W]
                    extrinsics = batch['RT'], # [B, T, 4, 4]
                    intrinsics = batch['camera_intrinsics'], # [B, T, 3,
                    condition_frames = batch['cond_frames'], # [B, N, C, H, W]
                    extrinsics_condition = batch['RT_cond'], # [B, N, 4, 4]
                    intrinsics_condition = batch['camera_intrinsics_cond'], # [B, N, 3, 3]
                    depth_maps = batch['depth_maps'], # [B, T, H, W]
                    depth_maps_condition = batch['depth_maps_cond'], # [B, N, H, W]
                    extrinsics_depth_maps = batch['RT_depth'], # [B, T, 4
                    intrinsics_depth_maps = batch['camera_intrinsics_depth'], # [B, T, 3, 3]
                    extrinsics_depth_maps_condition = batch['RT_depth_cond'], # [B, N, 4, 4]
                    intrinsics_depth_maps_condition= batch['camera_intrinsics_depth_cond'], # [B, N, 3, 3]
                    frame_stride = batch["frame_stride"],
                    caption = batch['caption'],
                    video_path = batch['video_path'],
                    steps=steps,
                    cfg_scale = cfg_scale,
                    trace_scale_factor=trace_scale_factor,
                    output_dir=output_path,
                    seed=seed,
                    enable_camera_condition=enable_camera_condition
                )

                gt_path = output_path / "raw" / "ground_truth.mp4"
                gen_path = output_path / "raw" / "generated.mp4"

                return gt_path, gen_path
                
            except Exception as e:
                print(f"Generation failed: {str(e)}")
                traceback.print_exc()
                return None, None

        def load_model(model_name: str, checkpoint: str):
            print(f"Loading model '{model_name}' with checkpoint '{checkpoint}'")
            config_path = model_map[model_name] / "config.yaml"
            width, height = 256, 256
            ckpt_path = checkpoint_map[model_map[model_name]][checkpoint]

            demo.load_model(
                config_file = config_path,
                width = width, height = height,
                ckpt_path = ckpt_path
            )
            print("Successfully loaded the model")
        
        def update_ckpt_dropdown(model_name: str) -> list:
            if model_name and model_name in ckpt_choices:
                return gr.update(choices=ckpt_choices[model_name], value=None)
            return gr.update(choices=[], value=None)

        def update_video_content(video_name: str):
            """Update video display, frame gallery, and selectors when video changes"""
            if not video_name:
                return [], gr.update(choices=[]), gr.update(choices=[])
            
            try:
                index = dataset.get_index_by_name(video_name)
                if index is None:
                    print(f"Video {video_name} not found in dataset")
                    return None, [], gr.update(choices=[]), gr.update(choices=[])
                    
                batch = dataset[index]
                
                # Get video for display (convert from [-1,1] to [0,1] and tensor format)
                video_tensor = batch['video']  # [3, T, H, W]
                video_array = ((video_tensor + 1.0) / 2.0).permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3] in [0,1]
                video_array = (video_array * 255).astype(np.uint8)  # Convert to uint8
                
                # Save video temporarily for Gradio display
                import tempfile
                import os
                
                # Create temp file in a way Gradio can access
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    temp_video_path = tmp_file.name
                
                # Use VidUtil to save the video
                temp_video = Video.fromArray(video_array, "THWC")
                temp_video.save(temp_video_path, fps=8)
                
                print(f"Saved temp video to: {temp_video_path}")
                print(f"File exists: {os.path.exists(temp_video_path)}")
                print(f"File size: {os.path.getsize(temp_video_path) if os.path.exists(temp_video_path) else 'N/A'}")
                
                # Create frame choices for selectors
                num_frames = video_array.shape[0]
                frame_choices = [str(i) for i in range(num_frames)]
                
                # Create frame gallery images
                frame_images = []
                for i in range(num_frames):  
                    frame = video_array[i]
                    frame_images.append((frame, f"Frame {i}"))
                
                return (
                    frame_images,  # frame_gallery
                    gr.update(choices=frame_choices, value=[]),  # reference_selector
                    gr.update(choices=frame_choices, value=[])   # frame_selector
                )
            except Exception as e:
                print(f"Failed to load video {video_name}: {e}")
                import traceback
                traceback.print_exc()
                return [], gr.update(choices=[]), gr.update(choices=[])

        
        ## Callbacks
        video_dropdown.change(
            fn=update_video_content,
            inputs=[video_dropdown],
            outputs=[frame_gallery, reference_selector, frame_selector]
        )
        model_dropdown.change(
            fn = update_ckpt_dropdown,
            inputs=[model_dropdown],
            outputs=[ckpt_dropdown]
        )
        load_btn.click(
            fn = load_model,
            inputs=[model_dropdown, ckpt_dropdown],
            outputs=[]
        )
        gen_btn.click(
            fn = generate,
            inputs=[model_dropdown, reference_selector, frame_selector, video_dropdown, trace_extract_ratio, trace_scale_factor, auto_reg_steps, enable_camera_condition, cfg_scale, frame_stride, steps, seed],
            outputs = [gt_video, gen_vid],
        )

    return app

def get_ip_addr():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    except:
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, default="./results", help="Path to the experiment directory holding the models.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on.")
    parser.add_argument("--machine", type=str, default="cvg28", help="Machine the gradio demo is being run on.")
    parser.add_argument("--output", type=Path, default="./visualization_results", help="Path to save evaluation results.")

    args = parser.parse_args()

    app = gradio_app(args.path, args.machine, args.output)
    print("Successfully initialized the gradio demo")
    print("Launching gradio demo!")
    app.queue(max_size=12)
    app.launch(max_threads=10, server_name=None, server_port=args.port)