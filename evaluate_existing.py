import os, sys
import argparse
from pathlib import Path

from VidUtil import Video
from VidUtil.metrics import FVD, SSIM, LPIPS, PSNR, MSE

from tqdm import tqdm
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate an existing path containing generated images")
    parser.add_argument("--path", "-p", required=True, type=Path, help="Path to the existing generated directory")
    args = parser.parse_args()

    video_paths = [p for p in args.path.iterdir() if p.is_dir()]

    print(f"Found {len(video_paths)} videos to be evaluated")

    fvd_metric = FVD()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    psnr_metric = PSNR()
    mse_metric = MSE()
    #import ipdb; ipdb.set_trace()

    for i, path in tqdm(enumerate(video_paths), total=len(video_paths), desc="Evaluating"):

        gen_path = path / "raw" / "generated.mp4"
        gt_path = path / "raw" / "ground_truth.mp4"
        
        if not os.path.exists(gen_path) or not os.path.exists(gt_path):
            print(f"Warning: video path is not complete: '{path}'")
            continue

        video_gt = Video.fromFile(gt_path)(format="TCHW", dtype="float32", data_range=(0,1), struct='torch')
        video_gen = Video.fromFile(gen_path)(format="TCHW", dtype="float32", data_range=(0,1), struct='torch')

        fvd_metric(video_gen, video_gt)
        ssim_metric(video_gen, video_gt)
        lpips_metric(video_gen, video_gt)
        psnr_metric(video_gen, video_gt)
        mse_metric(video_gen, video_gt)

    
    print(f"Evaluation finished!")
    fvd = fvd_metric.result
    ssim = ssim_metric.result
    lpips = lpips_metric.result
    psnr = psnr_metric.result
    mse = mse_metric.result

    result_dict = {
        "fvd": fvd,
        "ssim": ssim,
        "lpips": lpips,
        "psnr": psnr,
        "mse": mse,
    }

    import ipdb; ipdb.set_trace()
    print("\n".join([f" {k}: {float(v) if v is not None else 'none'}" for k, v in result_dict.items()]))
    with open(args.path / "new_results.yaml", "w") as f:
        yaml.dump(result_dict, f)






