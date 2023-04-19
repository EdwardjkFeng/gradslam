import torch
import numpy as np
import cv2 as cv
import time
import argparse
import open3d as o3d

__all__ = ["Visualizer"]

class Visualizer():
    """Dynamic gradSLAM visualization frontend"""
    def __init__(self, video, device="cuda:0"):
        torch.cuda.set_device(device)
        self.video = video
        self.cameras = {}
        self.background_points = {}
        self.foreground_objects = []
        self.warmup = 8
        self.scale = 1.0
        self.index = 0

        self.filter_thresh = 0.005

        self.vis = o3d.visualization.Visualizer()

    def animation_callback(self):
        cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with self.video.get_lock():
                t = self.video.counter.value
                

            if len(self.cameras) >= self.warmup:
                cam = self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            self.index += 1
            self.vis.poll_events()
            self.vis.update_renderer()


    def register_callback(self):
        self.vis.register_animation_callback(self.animation_callback)

    def run(self):
        self.vis.create_window(height = 540, width = 960)
        
        self.vis.run()
        self.vis.destroy_window()

    
