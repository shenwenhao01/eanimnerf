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
from lib.utils.vis_utils import generate_bar, write_pcd
from plyfile import PlyData

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose or cfg.aninerf_animation:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame

        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)                                 # joints位置 (24, 3)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))      # kinematic tree (24, )

        self.nrays = cfg.N_rand

        # read v_shaped
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
        '''
        Return:

        wxyz:    smpl vertices world coord (6890, 3)
        pxyz:    smpl vertices smpl coord 去除了全局RT
        A:       kinematic tree transformation (24, 4, 4)
        pbw:     observation space blend weights (25,72,36,25)
        Rh, Th:  smpl 在world coord全局RT
        '''
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)                        # smpl vertices world coord

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)                                    # smpl 在world coord全局变换
        Th = params['Th'].astype(np.float32)                

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)                          # smpl 顶点位置 去除了全局RT

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)     # kinematic tree transformation (24, 4, 4)

        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)                                            # observation space blend weights (25,72,36,25)

        return wxyz, pxyz, A, pbw, Rh, Th

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        wpts, ppts, A, pbw, Rh, Th = self.prepare_input(i)

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box          # (1024, )用来控制采样点near<far
        }

        # blend weight
        meta = {
            'A': A,                             # kinematic tree transformation (24, 4, 4)
            'pbw': pbw,                         # observation space blend weights (25,72,36,25)
            'tbw': self.tbw,                    # T-pose / canonical space blend weights (73, 71, 19, 25)
            'pbounds': pbounds,                 # smpl(pose) coord (world coord 去掉全局RT) bounds
            'wbounds': wbounds,                 # world coord bounds
            'tbounds': self.tbounds,                 # T-pose 的 bbox (2, 3)
            'tcenters': self.tpose_part_centers,     # tpose的part centers (24, 3)
            'tpart_bounds': self.tpose_part_bounds,  # tpose的part bounds (24, 3)
            'search_table': self.search_table,          # mapping: smpl part -> big part (24)
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'R': R,                             # human rotation matrix (3, 3)
            'Th': Th,                           # human translation matrix (1, 3)
            'H': H, 
            'W': W
        }
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose:
            if 'h36m' in self.data_root:
                latent_index = 0
            else:
                latent_index = cfg.num_train_frame - 1
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind                  # which camera
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
