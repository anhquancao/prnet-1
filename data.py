#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import pickle
import ruamel.yaml as yaml
import numpy as np
import random # np.random.choice doesn't like a list of tuples.
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import open3d as o3d


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 
    random_p2 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNet40, self).__init__()
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item, vis=True):
        # I don't understand how the ModelNet data works. Because somehow the point cloud is uniformly subsampled by simply taking the n last points.
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=self.num_subsampled_points)
        
        if vis:
            pcd1_v = o3d.utility.Vector3dVector(pointcloud1.T)
            pcd2_v = o3d.utility.Vector3dVector(pointcloud2.T)
            pcd1 = o3d.geometry.PointCloud(pcd1_v)
            pcd2 = o3d.geometry.PointCloud(pcd2_v)

            pcd1 = pcd1.paint_uniform_color([1, 0, 0])
            pcd2 = pcd2.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([pcd1, pcd2])

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]



class Bunny(Dataset):
    def __init__(self, num_subsampled_points=768, rot_factor=4, t_range=(-0.5, 0.5)):
        # self.data, self.label = load_data(partition)

        self.num_subsampled_points = num_subsampled_points
        self.rot_factor = rot_factor
        self.translation_range = t_range
        self.pcd = o3d.io.read_point_cloud("/home/grans/Documents/prnet2/bunny.ply")
        self.points = np.asarray(self.pcd.points)

        idcs = np.random.choice(np.arange(len(self.points)), size=1536, replace=False)
        self.points = self.points[idcs]
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)


    def pick_points(self):
        print("")
        print("1) Pick two points [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press q for close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self.pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    def dist(self, idcs):
        print(np.linalg.norm(self.points[idcs[0]] - self.points[idcs[1]]))

    def __getitem__(self, item=0, vis=False):
        print("Getting the item.")
        
        # picked_points = []
        # while len(picked_points) != 2:
        #     picked_points = self.pick_points()

        picked_points = np.random.randint(len(self.points), size=(2, ))
        self.dist(picked_points)

        knn = NearestNeighbors(n_neighbors=self.num_subsampled_points)
        knn.fit(self.points)
        pcd1_idcs = knn.kneighbors([self.points[picked_points[0]]], return_distance=False)
        pcd2_idcs = knn.kneighbors([self.points[picked_points[1]]], return_distance=False)

        overlap_idcs = np.intersect1d(pcd1_idcs, pcd2_idcs)

        pcd1 = self.points[pcd1_idcs][0] # (self.num_subsampled_points x 3)
        pcd2 = self.points[pcd2_idcs][0]

        ## This cancels all the stuff above.
        idcs = np.random.choice(np.arange(len(self.points)), size=self.num_subsampled_points, replace=False)
        pcd1 = pcd2 = self.points[idcs]

        if vis:
            pcd_c = np.zeros_like(self.points)

            pcd_c[:] = [1.0, 0, 0]
            pcd_c[overlap_idcs] = [0, 0, 1.0]
            pcd_c[picked_points] = [0, 0, 0]

            pcdo3d1 = o3d.geometry.PointCloud()
            pcdo3d1.points = o3d.utility.Vector3dVector(pcd1)
            pcdo3d1.colors = o3d.utility.Vector3dVector(pcd_c[pcd1_idcs][0])

            pcdo3d2 = o3d.geometry.PointCloud()
            pcdo3d2.points = o3d.utility.Vector3dVector(pcd2)

            pcd_c[:] = [0, 1.0, 0]
            pcd_c[overlap_idcs] = [0, 0, 1.0]
            pcd_c[picked_points] = [0, 0, 0]
            pcdo3d2.colors = o3d.utility.Vector3dVector(pcd_c[pcd2_idcs][0])

            o3d.visualization.draw_geometries([pcdo3d1, pcdo3d2])
        ## Apply rotation and translation here 

        # pointcloud = self.data[item][:self.num_points]
        # if self.partition != 'train':
        #     np.random.seed(item)

        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pcd1 = pcd1.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pcd2 = rotation_ab.apply(pcd2).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]


        #pcd1 = np.random.permutation(pcd1.T).T
        #pcd2 = np.random.permutation(pcd2.T).T

        #if self.subsampled:
        #    pcd1, pcd2 = farthest_subsample_points(pcd1, pcd2, num_subsampled_points=self.num_subsampled_points)


        if vis:
            pcd_c = np.zeros_like(self.points)

            pcd_c[:] = [1.0, 0, 0]
            pcd_c[overlap_idcs] = [0, 0, 1.0]

            pcdo3d1 = o3d.geometry.PointCloud()
            pcdo3d1.points = o3d.utility.Vector3dVector(pcd1.T)
            pcdo3d1.colors = o3d.utility.Vector3dVector(pcd_c[pcd1_idcs][0])

            pcdo3d2 = o3d.geometry.PointCloud()
            pcdo3d2.points = o3d.utility.Vector3dVector(pcd2.T)

            pcd_c[:] = [0, 1.0, 0]
            pcd_c[overlap_idcs] = [0, 0, 1.0]
            pcdo3d2.colors = o3d.utility.Vector3dVector(pcd_c[pcd2_idcs][0])

            o3d.visualization.draw_geometries([pcdo3d1, pcdo3d2])

        print("[%d, %d] Overlap: %f" % (picked_points[0], picked_points[1], (len(overlap_idcs)/self.num_subsampled_points)))

        

        return pcd1.astype('float32'), pcd2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return 1


class TLessModel(Dataset):
    def __init__(self, num_points=768, tlesspath='/home/grans/Documents/t-less_v2/', model='cadsub'):
        self.tlesspath = tlesspath
        if model == 'cad': 
            self.model = 'models_cad'
        elif model == 'cadsub':
            self.model = 'models_cad_subdivided'
        elif model == 'recon':
            self.model = 'models_reconst'
        else:
            raise NotImplementedError

        self.num_points = num_points

    def __getitem__(self, item=None, vis=True):
        if item is None:
            item = np.random.randint(30) + 1
        file_mask = os.path.join(self.tlesspath, self.model, 'obj_{:02d}.ply')
        print("Opening " + file_mask.format(item))
        mesh = o3d.io.read_triangle_mesh(file_mask.format(item))
        pcd = mesh.sample_points_poisson_disk(self.num_points)

        if vis:
            pcd = pcd.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([mesh, pcd])
        

    def __len__(self):
        return 1

class TLessScan(Dataset):
    def __init__(self, num_points=768, 
                tlesspath='/home/grans/Documents/t-less_v2/', 
                obj2scene='/home/grans/Documents/prnet2/obj2scenelist.pkl',
                scene2obj='/home/grans/Documents/prnet2/scene2objlist.pkl',
                scenes=None,
                objects=None,
                window_size=None):
        self.tlesspath = tlesspath
        self.num_points = num_points
        self.obj2scene = pickle.load(open(obj2scene, 'rb'))
        self.scene2obj = pickle.load(open(scene2obj, 'rb'))

        self.gt_mask = os.path.join(self.tlesspath, 'test_primesense', '{:02d}', 'gt.yml')
        self.info_mask = os.path.join(self.tlesspath, 'test_primesense', '{:02d}', 'info.yml')
        self.depth_image_mask = os.path.join(self.tlesspath, 'test_primesense', '{:02d}', 'depth', '{:04d}.png')

        if scenes is None:
            self.scenes = list(range(1, 21))
        if objects is None:
            self.objects = list(range(1, 31))

    def random_by_scene_id(self, scene_id):
        return np.random.choice(self.scene2obj[scene_id])

    def random_by_obj_id(self, obj_id):
        return np.random.choice(self.obj2scene[obj_id])
        
    def random(self):
        scene_id = np.random.choice(self.scenes)
        obj_id = np.random.choice(self.scene2obj[scene_id])
        return obj_id, scene_id

    def __getitem__(self, view_id=None, 
                        obj_id=None, 
                        scene_id=None, 
                        instance_idx=None, vis=True):
        if view_id is None: # This should probably be a specific view of a scene.
            view_id = np.random.randint(504) # Each scene has 504 images 0000.png to 503.png
        
        if obj_id is None:
            if scene_id is None:
                obj_id, scene_id = self.random()
            else:
                obj_id = self.random_by_scene_id(scene_id)
        else:
            scene_id = self.random_by_obj_id(obj_id)

        f = open(self.gt_mask.format(scene_id), 'r')
        gt = yaml.load(f, Loader=yaml.CLoader)
        f.close()

        f = open(self.info_mask.format(scene_id), 'r')
        info = yaml.load(f, Loader=yaml.CLoader)
        
        fx, _, cx, _, fy, cy, _, _, _ = np.array(info[view_id]['cam_K'])
        scale = np.array(info[view_id]['depth_scale'])

        depth_raw = o3d.io.read_image(self.depth_image_mask.format(scene_id, view_id))
        width, height = depth_raw.get_max_bound().astype('int')
        cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # We might have multiple instances of the same object in the scene, 
        # so we must get them all, and then select one of them.
        # TODO: If this process turns out to be slow, it might be useful to 
        # reconstruct the dictionary somehow
        obj_id_gts = []
        for obj_gt in gt[view_id]:
            if obj_gt['obj_id'] == obj_id:
                obj_id_gts.append(obj_gt)

        obj_id_gt = None
        if instance_idx is not None:
            obj_id_gt = obj_id_gts[instance_idx]
        else:
            instance_idx, obj_id_gt = random.choice(list(enumerate(obj_id_gts)))


        if vis:
            pcd = o3d.geometry.PointCloud().create_from_depth_image(depth_raw, 
                        cameraIntrinsics, 
                        depth_scale=1/scale)
            o3d.visualization.draw_geometries([pcd])

        

    def __len__(self):
            return 1

if __name__ == '__main__':
    # tlessmodel = TLessModel()
    # t = tlessmodel.__getitem__()
    tlscan = TLessScan()
    tlscan.__getitem__()

    #pts = 128
    # d = ModelNet40(num_points=pts,
    #                 num_subsampled_points=pts,
    #                 partition='test',
    #                 rot_factor=4)
    #d.__getitem__(0)
    print('hello world')
