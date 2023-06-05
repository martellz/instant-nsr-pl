import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering, ray_aabb_intersect


class VarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(VarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
    
    @property
    def inv_s(self):
        return torch.exp(self.variance * 10.0)

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.variance = VarianceNetwork(self.config.init_variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        self.bg_geometry = models.make(self.config.bg_geometry.name, self.config.bg_geometry)
        self.bg_texture = models.make(self.config.bg_texture.name, self.config.bg_texture)
        self.register_buffer('bg_aabb', torch.as_tensor([-self.config.bg_radius, -self.config.bg_radius, -self.config.bg_radius,
            self.config.bg_radius, self.config.bg_radius, self.config.bg_radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.bg_occupancy_grid = OccupancyGrid(
                roi_aabb = self.scene_aabb * 3.0,
                resolution = 128,
                contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.bg_render_step_size = 1.732 * 2 * self.config.bg_radius / self.config.bg_num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.bg_texture, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn)

        def occ_eval_fn_bg(x):
            density, _ = self.bg_geometry(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
            return density[...,None] * self.bg_render_step_size

        if self.training and self.config.grid_prune:
            self.bg_occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        # with torch.no_grad():
        # compute t_starts_bg and t_ends_bg
        intersect_min, intersect_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        rays_o_bg = rays_o.clone()
        # print("intersect_max shape: {}, rays_o shape: {}".format(intersect_max.shape, rays_o.shape))

        intersect_mask = intersect_max < 1e10
        print(torch.sum(intersect_mask))
        rays_o_bg[intersect_mask] = (rays_o + intersect_max[..., None] * rays_d)[intersect_mask]

        sdf_grad_samples = []

        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad = self.geometry(positions, with_grad=True, with_feature=False)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            return alpha[...,None]
        
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            sdf_grad_samples.append(sdf_grad)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            rgb = self.texture(feature, t_dirs, normal)
            return rgb, alpha[...,None]

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=alpha_fn,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

            # print("t_starts shape: {}, t_ends shape: {}, ray_indices shape: {}".format(t_starts.shape, t_ends.shape, ray_indices.shape))
            # t_starts shape: M x 1
            # t_ends shape: M x 1
            # ray_indices shape: M

        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_d.shape[0],
            rgb_alpha_fn=rgb_alpha_fn,
            render_bkgd=None,
        )

        # background
        def sigma_fn_bg(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o_bg[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.bg_geometry(positions)
            return density[...,None]

        def rgb_sigma_fn_bg(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o_bg[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.bg_geometry(positions)
            rgb = self.bg_texture(feature, t_dirs)
            return rgb, density[...,None]

        with torch.no_grad():
            ray_indices_bg, t_starts_bg, t_ends_bg = ray_marching(
                rays_o_bg, rays_d,
                scene_aabb=self.bg_aabb,
                grid=self.bg_occupancy_grid if self.config.grid_prune else None,
                sigma_fn=sigma_fn_bg,
                near_plane=None, far_plane=None,
                render_step_size=self.bg_render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        rgb_bg, opacity_bg, depth_bg = rendering(
            t_starts_bg,
            t_ends_bg,
            ray_indices_bg,
            n_rays=rays_d.shape[0],
            rgb_sigma_fn=rgb_sigma_fn_bg,
            render_bkgd=self.background_color
        )

        rgb = rgb * opacity + rgb_bg * (1.0 - opacity) * opacity_bg
        opacity = opacity + (1.0 - opacity) * opacity_bg
        rgb = rgb / (opacity + 1e-5)

        depth = torch.min(depth, depth_bg)

        sdf_grad_samples = torch.cat(sdf_grad_samples, dim=0)
        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

        rv = {
            'comp_rgb': rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts) + len(t_starts_bg)], dtype=torch.int32, device=rays.device),
            'comp_rgb_bg': rgb_bg,
            'opacity_bg': opacity_bg,
            'depth_bg': depth_bg
        }

        if self.training:
            rv.update({
                'sdf_grad_samples': sdf_grad_samples
            })

        return rv

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))

        losses.update(self.bg_geometry.regularizations(out))
        losses.update(self.bg_texture.regularizations(out))

        return losses

