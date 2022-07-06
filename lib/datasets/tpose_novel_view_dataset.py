import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.blend_utils import pts_sample_blend_weights
from plyfile import PlyData
from lib.utils import render_utils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        test_view = [3]
        view = cfg.training_view if split == 'train' else test_view
        self.num_cams = len(view)
        K, RT = render_utils.load_cam(ann_file)
        render_w2c = render_utils.gen_path(RT)

        i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][:cfg.num_train_frame *
                                          cfg.frame_interval]
        ])

        self.K = K[0]
        self.render_w2c = render_w2c
        img_root = 'data/render/{}'.format(cfg.exp_name)
        # base_utils.write_K_pose_inf(self.K, self.render_w2c, img_root)

        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        self.nrays = cfg.N_rand

        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        self.tvertices = np.load(vertices_path).astype(np.float32)              # T-pose下的人体vertices位置 (6890, 3)
        self.tbounds = if_nerf_dutils.get_bounds(self.tvertices)                # T-pose 的 bbox (2, 3)
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))                   # T-pose blend weights? (73, 71, 19, 25)
        self.tbw = tbw.astype(np.float32)
        self.tpose_part_centers, self.tpose_part_bounds, self.search_table = self.process_tpart()

    @staticmethod
    def bounds_from_vertices(vertices, scale=1.1, keep_scale=True):
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        center = (min_bounds + max_bounds)/2
        scale = ((max_bounds - min_bounds)/2 * scale)
        if keep_scale:
            scale = scale.max()
        min_bounds = center - scale
        max_bounds = center + scale
        bounds = np.stack([min_bounds, max_bounds])
        return bounds

    def process_tpart(self):
        import torch
        tvertices =  torch.from_numpy( self.tvertices[None] )
        tbounds = torch.from_numpy( self.tbounds[None] )
        tbw = torch.from_numpy( self.tbw[None] )
        tbw = pts_sample_blend_weights(tvertices, tbw, tbounds)
        tbw = tbw[0, :24]
        partid = torch.argmax(tbw, dim=0)                   # 6890 Tensor

        center_list, bounds_list, part_list, big_part_list = [], [], [], []
        for nj in range(24):
            part = tvertices[0, partid==nj, :].detach().cpu().numpy()
            part_list.append(part)
            if False:
                write_pcd(f"part-{i}.ply", part.numpy(), generate_bar(part.shape[0]))

        # concat into 11 big parts
        map_list = [
            [20, 22],               # l_hand
            [21, 23],               # r_hand
            [ 7, 10],               # l_foot
            [ 8, 11],               # r_foot
            [16, 18],               # l_arm
            [17, 19],               # r_arm
            [ 1,  4],               # l_leg
            [ 2,  5],               # r_leg
            [ 0,  3, 6],            # lower_torso
            [ 9, 12, 13, 14],       # upper_torso
            [15]                    # head
        ]
        search_table = np.zeros(24, dtype=int)
        for i in range(len(map_list)):
            assert isinstance(map_list[i], list), "Wrong mapping list"
            part_stack_list = []
            for partid in map_list[i]:
                part_stack_list.append(part_list[partid])
                search_table[partid] = i
            big_part_list.append(np.concatenate(part_stack_list, axis=0))
        print(search_table)
        assert len(big_part_list) == len(map_list)
        self.nBigPart = len(big_part_list)
        for ni in range( self.nBigPart ):
            big_part = big_part_list[ni]
            center = big_part.mean(axis=0, keepdims=True)
            bounds = self.bounds_from_vertices(big_part, scale=1.5)
            print('[Aninerf - Big Part {}] center = {}, bounds=({}) '.format(ni, center, bounds[1] - bounds[0]))
            center_list.append(center)
            bounds_list.append(bounds)
        center = np.stack(center_list)
        bounds = np.stack(bounds_list)
        return center, bounds, search_table

    def prepare_input(self, i):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1

        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th

    def get_mask(self, i):
        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, 'mask',
                                        im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(
                    self.data_root, im.replace('images', 'mask'))[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(
                    self.data_root, im.replace('images', 'mask'))[:-4] + '.jpg'
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

            msk = msk_cihp.astype(np.uint8)

            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[nv])

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        view_index = index
        latent_index = cfg.begin_ith_frame
        frame_index = cfg.begin_ith_frame * cfg.frame_interval

        wpts, ppts, A, pbw, Rh, Th = self.prepare_input(frame_index)

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        msks = self.get_mask(frame_index)

        # reduce the image resolution by ratio
        img_path = os.path.join(self.data_root, self.ims[0][0])
        img = imageio.imread(img_path)
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        K = self.K
        RT = self.render_w2c[index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)
        # ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
        #         RT, K, wbounds)

        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'pbw': pbw,
            'tbw': self.tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': self.tbounds,
            'tcenters': self.tpose_part_centers,     # tpose的part centers (24, 3)
            'tpart_bounds': self.tpose_part_bounds,  # tpose的part bounds (24, 3)
            'search_table': self.search_table,          # mapping: smpl part -> big part (24)
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index
        }
        ret.update(meta)

        meta = {'msks': msks, 'Ks': self.Ks, 'RT': self.RT, 'H': H, 'W': W}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.render_w2c)
