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

import os
import random
import json
import torch
from torch import load, save
from models.encoders import SharedIdentity
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.aabb_func = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            self.aabb_func = self.get_nerf_aabb_transform()
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if self.gaussians.qai_network is not None and self.gaussians._no_hyper:
                self.gaussians.qai_network.load_state_dict(load(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "qai_network.pt"), weights_only=True ))
            if self.gaussians.hypernetwork is not None:
                self.gaussians.hypernetwork.load_state_dict(load(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "hyper_model.pt"), weights_only=True))
            if self.gaussians.xyz_encoder is not None:
                self.gaussians.xyz_encoder.load_state_dict(load(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "xyz_encoder.pt"), weights_only=True))
            if self.gaussians.shared_encoder is not None and not isinstance(self.gaussians.shared_encoder, SharedIdentity):
                self.gaussians.shared_encoder.load_state_dict(load(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "shared_encoder.pt"), weights_only=True))
            with open(os.path.join(self.model_path, "vdgs_settings.json"), 'r') as file:
                vdgs_settings = json.load(file)
                self.gaussians.vdgs_type = vdgs_settings["vdgs_type"]
                self.gaussians.vdgs_operator = vdgs_settings["vdgs_operator"]
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        if self.aabb_func is None:
            self.aabb_func = self.get_large_scene_transform(scene_info.nerf_normalization)
    
    def get_nerf_aabb_transform(self):
        aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).float().cuda()
        def aabb_func(x):
            return (x - aabb[0]) / (aabb[1] - aabb[0])
        return aabb_func
    
    def get_large_scene_transform(self, nerf_normalization):
        translate = torch.from_numpy(nerf_normalization["translate"]).float().cuda()
        inner_radius = float(nerf_normalization["radius"])

        def transform_func(x):
            x_centered = (x - translate) / inner_radius
            
            mag = torch.linalg.norm(x_centered, dim=-1, keepdim=True)
            mask = mag <= 1.0
            
            x_contracted = torch.where(
                mask,
                x_centered,
                (2.0 - 1.0 / (mag + 1e-6)) * (x_centered / (mag + 1e-6))
            )
            
            x_final = (x_contracted + 2.0) / 4.0
            return x_final

        return transform_func

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.qai_network is not None and self.gaussians._no_hyper:
            save(self.gaussians.qai_network.state_dict(), os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "qai_network.pt"))
        if self.gaussians.hypernetwork is not None:
            save(self.gaussians.hypernetwork.state_dict(), os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "hyper_model.pt"))
        if self.gaussians.xyz_encoder is not None:
            save(self.gaussians.xyz_encoder.state_dict(), os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "xyz_encoder.pt"))
        if self.gaussians.shared_encoder is not None and not isinstance(self.gaussians.shared_encoder, SharedIdentity):
            save(self.gaussians.shared_encoder.state_dict(), os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "shared_encoder.pt"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]