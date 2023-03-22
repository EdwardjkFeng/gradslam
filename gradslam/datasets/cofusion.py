import os
import warnings
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from  ..geometry.geometryutils import relative_transformation
from torch.utils import data

from . import datautils

__all__ = ["Cofusion"]


class Cofusion(data.Dataset):
    r"""
    """
    
    def __init__(
        self,
        basedir: str,
        sequences: Union[tuple, str, None] = None,
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        height: int = 480,
        width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        *,
        return_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
        return_transform: bool = True,
        return_names: bool = True,
        return_object_mask: bool = True,
        return_object_label: bool = True,
    ):
        super(Cofusion, self).__init__()

        basedir = os.path.normpath(basedir)
        self.height = height
        self.width = width
        self.height_downsample_ratio = float(height) / 480
        self.width_downsample_ratio = float(width) / 640
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.return_transform = return_transform
        self.return_names = return_names
        self.return_object_mask = return_object_mask
        self.return_object_label = return_object_label

        self.load_poses = self.return_pose or self.return_transform

        if not isinstance(seqlen, int):
            raise TypeError("seqlen must be int. Got {0}.".format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError("stride must be int or None. Got {0}.".format(type(stride)))
        if not (isinstance(dilation, int) or dilation is None):
            raise TypeError("stride must be int or None. Got {0}.".format(type(dilation)))
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        if seqlen < 0:
            raise ValueError("seqlen must be positive. Got {0}.".format(seqlen))
        if dilation < 0:
            raise ValueError("dilation must be positive. Got {0}.".format(dilation))
        if stride < 0:
            raise ValueError("stride must be positive. Got {0}.".format(stride))

        if not (isinstance(start, int) or start is None):
            raise TypeError("start must be int or None. Got {0}.".format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError("end must be int or None. Got {0}.".format(type(end)))
        start = start if start is not None else 0
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(start))
        if not (end is None or end > start):
            raise ValueError(
                "end ({0}) must be None or greater than start ({1})".format(end, start)
            )

        # prepocess trajectories to be a tuple or None
        # TODO: check basedir of traj
        valid_trajectory_dirs = [
            f for f in os.listdir(basedir)
            if os.path.isdir(os.path.join(basedir, f))
            and f[-5:] == "-full"
        ]
        if len(valid_trajectory_dirs) == 0:
            msg = "basedir ({0}) should contain trajectory folders with the following naming ".format(
                basedir
            )
            msg += 'convention: "X-full". Found 0 folders with this naming convention.'
            raise ValueError(msg)
        
        if isinstance(sequences, str):
            if os.path.isfile(sequences):
                with open(sequences, "r") as f:
                    sequences = tuple(f.read().split("\n"))
                valid_trajectory_dirs = list(sequences)
            else:
                raise ValueError(
                    "incorrect filename: {} doesn't exist".format(sequences)
                )
        elif not (sequences is None or isinstance(sequences, tuple)):
            msg = '"sequences" should either be path to .txt file or tuple of trajectory names or None, '
            msg += " but was of type {0} instead"
            raise TypeError(msg.format(type(sequences)))
        if isinstance(sequences, tuple):
            if len(sequences) == 0:
                raise ValueError(
                    '"sequences" must have at least one element. Got len(sequences) = 0'
                )
            msg = '"sequences" should only contain trajectory folder names of the following convention: '
            msg += '"X-full". It contained: {0}.'
            for t in sequences:
                if not (t[-5:] == "-full"):
                    raise ValueError(msg.format(t))
            valid_trajectory_dirs = list(sequences)

        # Check if CoFusion folder structure correct: If sequences is not None, should contain all sequence paths.
        # Should also contain at least one sequence path
        sequence_paths = []
        dirmsg = "Cofusion folder should look something like:\n\n| ├── basedir\n"
        dirmsg += "| │   ├── X-full\n| │   │   ├── colour/\n"
        dirmsg += "| │   │   ├── depth_noise/\n| │   │   ├── depth_original/\n"
        dirmsg += "| │   │   ├── mask_colour/\n| │   │   ├── mask_id/\n"
        dirmsg += "| │   │   ├── trajectories/\n"
        dirmsg += "| │   │   ├── calibration.txt\n"
        dirmsg += "| │   │   └── color-X.mp4"
        for item in os.listdir(basedir):
            if (
                os.path.isdir(os.path.join(basedir, item))
                and item in valid_trajectory_dirs
            ):
                sequence_paths.append(os.path.join(basedir, item))
        if len(sequence_paths) == 0:
            raise ValueError(
                'Incorrect folder structure in basedir ("{0}"). '.format(basedir)
                + dirmsg
            )
        if sequences is not None and len(sequence_paths) != len(sequence_paths):
            msg = '"sequences" contains sequences not available in basedir:\n'
            msg += "sequences contains: " + ", ".join(sequences) + "\n"
            msg += (
                "basedir contains: "
                + ", ".join(list(map(os.path.basename, sequence_paths)))
                + "\n"
            )
            raise ValueError(msg.format(basedir) + dirmsg)
        
        # TODO: Check, get association and pose file paths
        posesfiles = []
        for sequence_path in sequence_paths:
            if self.load_poses:
                posesfile = os.path.join(
                    sequence_path, "trajectories/", "gt-cam-0.txt"
                )
                if not os.path.isfile(posesfile):
                    msg = 'Missing ground truth poses file ("{0}") in {1}. '.format(
                        posesfile, basedir
                    )
                    raise ValueError(msg + dirmsg)
                posesfiles.append(posesfile)

        

        # Get a list of all color, depth, pose, label and intrinsics files.
        colorfiles, depthfiles, poses, framenames = [], [], [], []
        maskfiles, labelfiles, intrinsicfiles = [], [], []
        idx = np.arange(seqlen) * (dilation + 1)
        for file_num, posefile in enumerate(posesfiles): # Iterate over all sequences
            parentdir = os.path.dirname(posefile)[:-12]
            splitpath = posefile.split(os.sep)
            trajectory_name = splitpath[-3]
            if sequences is not None:
                if trajectory_name not in sequences:
                    continue

            traj_colorfiles, traj_depthfiles = [], []
            traj_maskfiles, traj_labelfiles, traj_intrinsicfile = [], [], []
            traj_poses, traj_framenames = [], []
            with open(posefile, "r") as f:
                lines = f.readlines()
                if self.end is None:
                    end = len(lines)
                if end > len(lines):
                    msg = "end was larger than number of frames in trajectory: {0} > {1} (trajectory: {2})"
                    warnings.warn(msg.format(end, len(lines), trajectory_name))
                lines = lines[start:end]
                if self.load_poses:
                    len_posesfile = sum(1 for line in f)


            for line_num, line in enumerate(lines): # Iterate files with current sequence
                line = line.strip().split()
                # TODO: check if reading form files is correct
                msg = "Incorrect reading from Cofusion associations"
                # Read rgb image file paths (PNG)
                traj_colorfiles.append(os.path.normpath(os.path.join(parentdir, "colour/Color{0:04d}.png".format(int(line[0])))))
                # Read depth image file paths (EXR)
                traj_depthfiles.append(os.path.normpath(os.path.join(parentdir, "depth_noise/Depth{0:04d}.exr".format(int(line[0])))))
                # Read mask image file paths (PNG)
                traj_maskfiles.append(os.path.normpath(os.path.join(parentdir, "mask_colour/Mask{0:04d}.png".format(int(line[0])))))
                # Read label file paths (PNG)
                traj_labelfiles.append(os.path.normpath(os.path.join(parentdir, "mask_id/Mask{0:04d}.png".format(int(line[0])))))

                if self.load_poses:
                    traj_poses.append(np.asarray(line[1:]))
                traj_framenames.append(
                    os.path.join(trajectory_name , line[0])
                )
            
            traj_len = len(traj_colorfiles)
            for start_ind in range(0, traj_len, stride):
                if (start_ind + idx[-1]) >= traj_len:
                    break
                inds = start_ind + idx
                colorfiles.append([traj_colorfiles[i] for i in inds])
                depthfiles.append([traj_depthfiles[i] for i in inds])
                maskfiles.append([traj_maskfiles[i] for i in inds])
                labelfiles.append([traj_labelfiles[i] for i in inds])
                framenames.append(", ".join([traj_framenames[i] for i in inds]))
                if self.load_poses:
                    poses.append([traj_poses[i] for i in inds])
            
            traj_intrinsicfile = os.path.normpath(os.path.join(parentdir, "calibration.txt"))

            intrinsicfiles.append(traj_intrinsicfile)
        
        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles
        self.maskfiles = maskfiles
        self.labelfiles = labelfiles
        self.poses = poses
        self.framenames = framenames

        # Camera intrinisics matrix for ICL dataset
        # TODO: read intrinsics from calibration files
        # if trajectory_name == 'room4-full':
        #     intrinsics = torch.tensor(
        #         [[360, 0, 320, 0], [0, 360, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        #     ).float()
        # elif trajectory_name == 'car4-full':
        #     intrinsics = torch.tensor(
        #         [[564.3, 0, 480, 0], [0, 564.3, 270, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        #     ).float()
        intrinsics = self._read_calibration(intrinsicfiles[0])
        
        self.intrinsics = datautils.scale_intrinsics(
            intrinsics, self.height_downsample_ratio, self.width_downsample_ratio
        ).unsqueeze(0)

        # Scaling factor for depth images
        self.scaling_factor = 1.0

    def __len__(self):
        r"""Return the length of the dataset."""
        return self.num_sequences
    
    def __getitem__(self, idx: int):
        r"""Return the data from the sequence at index idx.
        
        Return:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequnece of depth images of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            transform_seq (torch.Tensor): Sequence of transformations between each frame in the sequence and the previous frame. Transformations are w.r.t. the first frame in the sequence having identity pose (relative transformations with first frmae's pose as the reference transformation). First transformation in the sequence will always be 'torch.eye(4)'.
            intrinsics (torch.Tensor): Intrinsics for the current sequence framename (str): Name of the frame
            
        Shape::
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth, mask, pose, label and intrinsics info.
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]
        mask_seq_path = self.maskfiles[idx]
        label_seq_path = self.labelfiles[idx]
        pose_pointquat_seq = self.poses[idx] if self.load_poses else None
        framename = self.framenames[idx]

        color_seq, depth_seq, mask_seq, pose_seq, label_seq = [], [], [], [], []
        for i in range(self.seqlen):
            color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
            mask = np.asarray(imageio.imread(mask_seq_path[i]), dtype=np.uint8)
            label = np.asarray(imageio.imread(label_seq_path[i]), dtype=np.uint8)
            # color = self._filter_moving_objects(color, mask)
            color = self._preprocess_color(color)
            color = torch.from_numpy(color)
            color_seq.append(color)

            if self.return_depth:
                depth = np.asarray(imageio.imread(depth_seq_path[i]), dtype=float)
                if len(depth.shape) > 2:
                    depth = depth[:, :, 0]
                # depth = self._filter_moving_objects(depth, mask)
                depth = self._preprocess_depth(depth)
                depth = torch.from_numpy(depth)
                depth_seq.append(depth)
            
            if self.return_object_mask:
                mask = self._preprocess_color(mask)
                mask_seq.append(torch.from_numpy(mask))
            
            # TODO: return label
            if self.return_object_label:
                label = self._preprocess_color(label)
                label_seq.append(torch.from_numpy(label))

        if self.load_poses:
            poses = self._homogenPoses(pose_pointquat_seq)
            pose_seq = [torch.from_numpy(pose) for pose in poses]

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)

        if self.return_depth:
            depth_seq = torch.stack(depth_seq, 0).float()
            output.append(depth_seq)

        if self.return_intrinsics:
            intrinsics = self.intrinsics
            output.append(intrinsics)
        
        if self.return_pose:
            pose_seq = torch.stack(pose_seq, 0).float()
            pose_seq = self._preprocess_poses(pose_seq)
            output.append(pose_seq)

        if self.return_transform:
            transform_seq = datautils.poses_to_transforms(poses)
            transform_seq = [torch.from_numpy(x).float() for x in transform_seq]
            transform_seq = torch.stack(transform_seq, 0).float()
            output.append(transform_seq)

        if self.return_names:
            output.append(framename)
        
        if self.return_object_mask:
            mask_seq = torch.stack(mask_seq, 0).to(dtype=torch.uint8)
            output.append(mask_seq)
        
        if self.return_object_label:
            label_seq = torch.stack(label_seq, 0).to(dtype=torch.uint8)
            output.append(label_seq)

        return tuple(output)

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _dilate_mask(self, mask: np.ndarray):
        r"""Dilate the masked area of dynamic objects"""
        kernel = np.ones((2, 2), np.uint8)
        mask_dilation = cv2.dilate(mask, kernel, iterations=1)
        return mask_dilation
    
    def _erode_mask(self, mask: np.ndarray):
        r"""Erode the masked area of dynamic objects"""
        kernel = np.ones((2, 2), np.uint8)
        mask_dilation = cv2.erode(mask, kernel, iterations=1)
        return mask_dilation

    def _filter_moving_objects(self, image: np.ndarray, mask: np.ndarray):
        r"""Filter moving objects from RGB image with mask.

        Args:
            image (np.ndarray): Raw RGB image or Raw depth image
            mask (np.ndarray): Mask for moving objects #TODO: generate mask for dynamic objects
        
        Returns:
            static_image (np.ndarray): RGB image with moving objects in black
        """
        # mask = self._dilate_mask(mask)
        # mask = self._erode_mask(mask)
        mask_filter = np.sum(mask, axis=2) == 0
        if len(image.shape) == 3:
            image[mask_filter, :] = [0, 0, 0]
        else:
            image[mask_filter] = 0
        return image

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.scaling_factor

    def _read_calibration(self, intrinsicfile: str):
        r"""Read camera intrinsic parameters from calibration.txt.

        Args:
            intrinsicfile (str): path to calibration.txt
        
        Returns:
            Output (torch.Tensor): K matrix

        Shape:
            - Output: math:`(4, 4)` 
        """
        with open(intrinsicfile, "r") as f:
            line = f.readline()
            line = line.strip().split()

        return torch.tensor(
                [[float(line[0]), 0, float(line[2]), 0], 
                 [0, float(line[1]), float(line[3]), 0], 
                 [0, 0, 1, 0], 
                 [0, 0, 0, 1]]
            ).float()

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        # return relative_transformation(
        #     poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1), poses,
        # )
        return relative_transformation(
            poses[0].unsqueeze(0).expand(poses.shape[0], -1, -1), poses,
        )

    def _homogenPoses(self, poses_point_quaternion):
        r"""Loads poses from groundtruth pose text files and returns the poses
        as a list of numpy arrays.

        Args:
            pose_path (str): The path to groundtruth pose text file.
            start_lines (list of ints):

        Returns:
            poses (list of np.array): List of ground truth poses in
                    np.array format. Each np.array has a shape of [4, 4] if
                    homogen_coord is True, or a shape of [3, 4] otherwise.
        """
        return [
            datautils.pointquaternion_to_homogeneous(pose)
            for pose in poses_point_quaternion
        ]
