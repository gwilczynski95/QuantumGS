#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import wandb

def apply_vdgs_operation(operator, factor, value):
    if operator == "mul":
        return factor * value
    elif operator == "add":
        return factor + value
    elif operator == "raw":
        return factor
    else:
        raise ValueError("Wrong VDGS operator. We support only mul|add")
    
def process_vdgs(pipe, color, opacity, factor, max_sh_degree):
    vdgs_operations = {
        "opacity": lambda factor, value: apply_vdgs_operation(pipe.vdgs_operator, factor, value),
        "color": lambda factor, value: apply_vdgs_operation(pipe.vdgs_operator, factor, value),
        "both": lambda factor, values: (apply_vdgs_operation(pipe.vdgs_operator, factor[0], values[0]), 
                                        apply_vdgs_operation(pipe.vdgs_operator, factor[1], values[1]))
    }

    if pipe.vdgs_type in vdgs_operations:
        color_factor_size = (max_sh_degree + 1)**2 * 3
        if pipe.vdgs_type == "both":
            color_factor, opacity_factor = torch.split(factor, [color_factor_size, 1], dim=1)
            color_factor = torch.reshape(color_factor, (-1, color_factor_size // 3, 3))
            color, opacity = vdgs_operations[pipe.vdgs_type]([color_factor, opacity_factor], [color, opacity])
        else:
            if pipe.vdgs_type == "opacity":
                value = opacity
            else:
                factor = torch.reshape(factor, (-1, color_factor_size // 3, 3))
                value = color
            result = vdgs_operations[pipe.vdgs_type](factor, value)
            if pipe.vdgs_type == "opacity":
                opacity = result
            else:
                color = result
    else:
        raise ValueError("Unsupported VDGS type")

    return color, opacity


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, iteration = None, use_wandb = False, aabb_func=lambda x: x):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if pc.hypernetwork or pc._no_hyper:
        cc = viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp = (pc.get_xyz - cc)
        dir_pp_normalized = F.normalize(dir_pp, p=2, dim=-1, eps=1e-3)
        assert torch.isnan(dir_pp_normalized).any().item() == False, "NaN in dir_pp_normalized"
        shared_features = pc.shared_encoder(dir_pp_normalized)
        assert torch.isnan(shared_features).any().item() == False, "NaN in shared features"
        # Add aabb [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]] to normalize means3D
        normalized_means3D = aabb_func(means3D)
        if pc._no_hyper:
            shared_xyz_features = pc.xyz_encoder(normalized_means3D)
            assert torch.isnan(shared_xyz_features).any().item() == False, "NaN in shared xyz features"
            shared_features = shared_features + shared_xyz_features
            shared_features = F.normalize(shared_features, p=2, dim=-1, eps=1e-3)
            assert torch.isnan(shared_features).any().item() == False, "NaN in shared features"
            weights = None
        else:
            weights = pc.hypernetwork(normalized_means3D)
        factor = pc.qai_network(shared_features, batched_weights=weights)
        shs, opacity = process_vdgs(pc, shs, opacity, factor, pc.max_sh_degree)
        if pc.vdgs_type in ["both", "opacity"]:
            pc.current_opacity = opacity.detach()
        if use_wandb:
            wandb.log({"train/features_mean": shared_features.mean(), "iteration": iteration})
            wandb.log({"train/features_std": shared_features.std(), "iteration": iteration})
            wandb.log({"train/features_min": shared_features.min(), "iteration": iteration})
            wandb.log({"train/features_max": shared_features.max(), "iteration": iteration})
            wandb.log({"train/factor_mean": factor.mean(), "iteration": iteration})
            wandb.log({"train/factor_std": factor.std(), "iteration": iteration})
            wandb.log({"train/factor_min": factor.min(), "iteration": iteration})
            wandb.log({"train/factor_max": factor.max(), "iteration": iteration})
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
