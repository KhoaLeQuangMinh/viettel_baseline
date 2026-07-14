import torch
import os
import pandas as pd
from tqdm import tqdm
from os import makedirs
from PIL import Image
import numpy as np
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_test_poses(model_path, csv_path, iteration, gaussians, pipeline, background, train_test_exp, separate_sh):
    # Load csv
    df = pd.read_csv(csv_path)
    
    # We will save rendered images to: {model_path}/test_renders/ours_{iteration}
    render_path = os.path.join(model_path, "test_renders", "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    
    print(f"Found {len(df)} test camera poses in {csv_path}. Rendering...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rendering test poses"):
        image_name = row['image_name']
        image_stem = os.path.splitext(image_name)[0]
        
        qw = row['qw']
        qx = row['qx']
        qy = row['qy']
        qz = row['qz']
        tx = row['tx']
        ty = row['ty']
        tz = row['tz']
        fx = row['fx']
        fy = row['fy']
        cx = row['cx']
        cy = row['cy']
        width = int(row['width'])
        height = int(row['height'])
        
        # Convert quaternion to rotation matrix
        qvec = np.array([qw, qx, qy, qz])
        # Transpose rotation because that is how 3DGS/GLM stores it internally
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array([tx, ty, tz])
        
        # Calculate field of views (FoV)
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        
        # Create a dummy image for Camera initialization
        dummy_image = Image.new("RGB", (width, height), (0, 0, 0))
        
        # Instantiate Camera object
        cam = Camera(
            resolution=(width, height),
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=dummy_image,
            invdepthmap=None,
            image_name=image_stem,
            uid=idx,
            data_device="cuda",
            train_test_exp=train_test_exp,
            is_test_dataset=True,
            is_test_view=True
        )
        
        # Render
        rendering = render(cam, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        
        # Save image
        import torchvision
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{image_name}"))
        
    print(f"\n[SUCCESS] Rendered test images saved to: {render_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--csv_path", required=True, type=str, help="Path to test_poses.csv")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Load model
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Find iteration
    from utils.system_utils import searchForMaxIteration
    if args.iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
        
    print("Loading trained model at iteration {}".format(loaded_iter))
    gaussians.load_ply(os.path.join(dataset.model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"), 
                       dataset.train_test_exp)
                       
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_test_poses(
        model_path=dataset.model_path,
        csv_path=args.csv_path,
        iteration=loaded_iter,
        gaussians=gaussians,
        pipeline=pipeline.extract(args),
        background=background,
        train_test_exp=dataset.train_test_exp,
        separate_sh=SPARSE_ADAM_AVAILABLE
    )
