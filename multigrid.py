from typing import Any, List

import numpy as np
from scipy.sparse.linalg import spsolve, splu

from mesh import UniformMesh
from ndassemble_cube3d import NDAssembleCube3D
from ndintergrid_cube3d import NDIntergridCube3D


class MG:
    smoother_index_vertex: List[List[dict]]
    smoother_index_edge: List[List[dict]]
    a_ii_solver: List[Any]
    s_gg_solver_f: List[Any]
    s_gg_solver_e: List[Any]

    def __init__(self, level, cycle, num_presmoothing, num_postsmoothing, smoothing_method, damping_factor, alpha=1.0,
                 beta=1.0, mesh_name='cube'):
        self.level = level

        self.alpha = alpha
        self.beta = beta

        self.mesh = [UniformMesh(2 ** (i + 1), mesh_name) for i in np.arange(self.level + 1)]
        self.nd = [NDAssembleCube3D(self.mesh[i], self.alpha, self.beta) for i in np.arange(self.level + 1)]
        self.h = [mesh.get_h() for mesh in self.mesh]

        self.boundary = [self.mesh[i].get_boundary_edge() for i in np.arange(self.level + 1)]
        self.interior = [np.setdiff1d(np.arange(self.mesh[i].get_edge_size()), self.boundary[i]) for i in
                         np.arange(self.level + 1)]

        self.mass_matrix = [self.nd[i].mass_assemble() for i in np.arange(self.level + 1)]
        self.curlcurl_matrix = [self.nd[i].curlcurl_assemble() for i in np.arange(self.level + 1)]
        self.stiffness_matrix = [self.mass_matrix[i] + self.curlcurl_matrix[i] for i in
                                 np.arange(self.level + 1)]
        self.stiffness_matrix_interior = [
            self.stiffness_matrix[i].tocsr()[self.interior[i], :].tocsc()[:, self.interior[i]].tocsr() for i in
            np.arange(self.level + 1)]

        self.intergrid_transfer = [NDIntergridCube3D(mesh_coarse=self.mesh[i], mesh_fine=self.mesh[i + 1]) for i in
                                   np.arange(self.level)]
        self.c2f = [self.intergrid_transfer[i].get_nd_intergrid_transfer_c2f() for i in np.arange(self.level)]
        self.f2c = [self.intergrid_transfer[i].get_nd_intergrid_transfer_f2c() for i in np.arange(self.level)]
        self.substructure = [self.intergrid_transfer[i].get_substructure() for i in np.arange(self.level)]

        self.cycle = cycle
        self.num_presmoothing = num_presmoothing
        self.num_postsmoothing = num_postsmoothing
        self.smoothing_method = smoothing_method
        self.damping_factor = damping_factor
        self.num_coarse_grid_correction = None

        if self.cycle == 'v-cycle':
            self.num_coarse_grid_correction = 1
        elif self.cycle == 'w-cycle':
            self.num_coarse_grid_correction = 2
        else:
            print('unsupported cycle')
            exit(1)

        self.smoother_index_edge = [[] for _ in np.arange(self.level + 1)]
        self.smoother_index_vertex = [[] for _ in np.arange(self.level + 1)]
        self.smoother_index_sc = [dict() for _ in np.arange(self.level + 1)]
        self.smoother_index_face_gamma_ov = [[] for _ in np.arange(self.level + 1)]
        self.smoother_index_edge_gamma_ov = [[] for _ in np.arange(self.level + 1)]
        self.matrices = [dict() for _ in np.arange(self.level + 1)]
        self.stiff_sub = [[] for _ in np.arange(self.level + 1)]
        self.sc_solver = [[] for _ in np.arange(self.level + 1)]
        self.a_ii_solver = [[] for _ in np.arange(self.level + 1)]
        self.s_gg_solver_f = [[] for _ in np.arange(self.level + 1)]
        self.s_gg_solver_e = [[] for _ in np.arange(self.level + 1)]

        for i in np.arange(self.level):
            self.smoother_setting(i + 1)

    def set_num_preprocessing(self, num_presmoothing):
        self.num_presmoothing = num_presmoothing

    def set_num_postprocessing(self, num_postprocessing):
        self.num_postsmoothing = num_postprocessing

    def smoother_setting(self, k):
        print('smoother setting for level ' + str(k))
        self.smoother_setting_edge(k)
        self.smoother_setting_vertex(k)
        self.smoother_setting_sc(k)
        self.smoother_setting_interior(k)
        if self.smoothing_method == 'face_edge_ov':
            self.smoother_setting_edge_ov(k)

    def smoother_setting_edge(self, k):
        coarse_face_size = self.mesh[k - 1].get_face_size()
        coarse_boundary_face = self.mesh[k - 1].get_boundary_face()
        coarse_interior_face = np.setdiff1d(np.arange(coarse_face_size), coarse_boundary_face)
        coarse_interior_edge = self.interior[k - 1]
        coarse_to_fine_edge = self.substructure[k - 1].get_coarse_to_fine_edge()
        coarse_element_interior_edge = self.substructure[k - 1].get_coarse_element_interior_edge()
        neighboring_coarse_elements = self.mesh[k - 1].get_edge()['element']
        coarse_face_interior_edge = self.substructure[k - 1].get_coarse_face_interior_edge()
        neighboring_coarse_faces = self.mesh[k - 1].get_edge()['face']
        index_sub = []
        for i in coarse_interior_edge:
            index_sub_i = dict(
                {'interior': np.array([], dtype=int), 'face': np.array([], dtype=int), 'edge': np.array([], dtype=int),
                 'face_index_interior': np.array([], dtype=int)})
            index_sub_i['edge'] = np.append(index_sub_i['edge'], coarse_to_fine_edge[i])
            coarse_elements_indices = neighboring_coarse_elements[i]
            for coarse_element in coarse_elements_indices:
                if coarse_element == -1:
                    continue
                index_sub_i['interior'] = np.append(index_sub_i['interior'],
                                                    coarse_element_interior_edge[coarse_element])
            coarse_face_indices = neighboring_coarse_faces[i]
            for coarse_face in coarse_face_indices:
                if coarse_face == -1:
                    continue
                index_sub_i['face'] = np.append(index_sub_i['face'], coarse_face_interior_edge[coarse_face])
                index_sub_i['face_index_interior'] = np.append(index_sub_i['face_index_interior'],
                                                               np.where(coarse_interior_face == coarse_face))
            index_sub.append(index_sub_i)
        self.smoother_index_edge[k] = index_sub

    def smoother_setting_vertex(self, k):
        coarse_face_size = self.mesh[k - 1].get_face_size()
        coarse_boundary_face = self.mesh[k - 1].get_boundary_face()
        coarse_interior_face = np.setdiff1d(np.arange(coarse_face_size), coarse_boundary_face)
        coarse_interior_edge = self.interior[k - 1]
        coarse_vertex_size = self.mesh[k - 1].get_vertex_size()
        coarse_boundary_vertex = self.mesh[k - 1].get_boundary_vertex()
        coarse_interior_vertex = np.setdiff1d(np.arange(coarse_vertex_size), coarse_boundary_vertex)

        coarse_to_fine_edge = self.substructure[k - 1].get_coarse_to_fine_edge()
        coarse_element_interior_edge = self.substructure[k - 1].get_coarse_element_interior_edge()
        neighboring_coarse_elements = self.mesh[k - 1].get_vertex()['element']
        coarse_face_interior_edge = self.substructure[k - 1].get_coarse_face_interior_edge()
        neighboring_coarse_faces = self.mesh[k - 1].get_vertex()['face']
        neighboring_coarse_edges = self.mesh[k - 1].get_vertex()['edge']

        index_sub = []
        for i in coarse_interior_vertex:
            index_sub_i = dict(
                {'interior': np.array([], dtype=int), 'face': np.array([], dtype=int), 'edge': np.array([], dtype=int),
                 'face_index_interior': np.array([], dtype=int), 'edge_index_interior': np.array([], dtype=int)})
            coarse_elements_indices = neighboring_coarse_elements[i]
            for coarse_element in coarse_elements_indices:
                if coarse_element == -1:
                    continue
                index_sub_i['interior'] = np.append(index_sub_i['interior'],
                                                    coarse_element_interior_edge[coarse_element])
            coarse_face_indices = neighboring_coarse_faces[i]
            for coarse_face in coarse_face_indices:
                if coarse_face == -1:
                    continue
                index_sub_i['face'] = np.append(index_sub_i['face'], coarse_face_interior_edge[coarse_face])
                index_sub_i['face_index_interior'] = np.append(index_sub_i['face_index_interior'],
                                                               np.where(coarse_interior_face == coarse_face))
            coarse_edge_indices = neighboring_coarse_edges[i]
            for coarse_edge in coarse_edge_indices:
                if coarse_edge == -1:
                    continue
                index_sub_i['edge'] = np.append(index_sub_i['edge'], coarse_to_fine_edge[coarse_edge])
                index_sub_i['edge_index_interior'] = np.append(index_sub_i['edge_index_interior'],
                                                               np.where(coarse_interior_edge == coarse_edge))
            index_sub.append(index_sub_i)
        self.smoother_index_vertex[k] = index_sub

    def smoother_setting_sc(self, k):
        index = dict(
            {'interior': np.array([], dtype=int), 'face': np.array([], dtype=int), 'edge': np.array([], dtype=int)})
        matrices = dict()
        coarse_element_interior_edge = self.substructure[k - 1].get_coarse_element_interior_edge()
        for element_index in coarse_element_interior_edge:
            index['interior'] = np.append(index['interior'], element_index)
        coarse_face_interior_edge = self.substructure[k - 1].get_coarse_face_interior_edge()
        coarse_face_size = self.mesh[k - 1].get_face_size()
        coarse_boundary_face = self.mesh[k - 1].get_boundary_face()
        coarse_interior_face = np.setdiff1d(np.arange(coarse_face_size), coarse_boundary_face)
        for i in coarse_interior_face:
            index['face'] = np.append(index['face'], coarse_face_interior_edge[i])
        coarse_interior_edge = self.interior[k - 1]
        coarse_to_fine_edge = self.substructure[k - 1].get_coarse_to_fine_edge()
        for i in coarse_interior_edge:
            index['edge'] = np.append(index['edge'], coarse_to_fine_edge[i])

        interior_index = index['interior']
        face_index = index['face']
        edge_index = index['edge']
        gamma_index = np.append(face_index, edge_index)

        a_ii = self.stiffness_matrix[k].tocsr()[interior_index, :].tocsc()[:, interior_index].tocsr()
        a_ig = self.stiffness_matrix[k].tocsr()[interior_index, :].tocsc()[:, gamma_index].tocsr()
        a_gi = self.stiffness_matrix[k].tocsr()[gamma_index, :].tocsc()[:, interior_index].tocsr()
        a_gg = self.stiffness_matrix[k].tocsr()[gamma_index, :].tocsc()[:, gamma_index].tocsr()

        matrices['a_ii'] = a_ii
        matrices['a_ig'] = a_ig
        matrices['a_gi'] = a_gi
        matrices['a_gg'] = a_gg

        self.a_ii_solver[k] = splu(a_ii.tocsc())

        s_gg = a_gg - np.dot(a_gi, spsolve(a_ii.tocsc(), a_ig.tocsc()))
        matrices['s_gg'] = s_gg

        self.matrices[k] = matrices
        self.smoother_index_sc[k] = index

        if self.smoothing_method == 'edge_gamma_is':
            coarse_interior_edge_size = len(self.interior[k - 1])

            for i in np.arange(coarse_interior_edge_size):
                face_index_sub_g = self.smoother_index_edge[k][i]['face']
                edge_index_sub_g = self.smoother_index_edge[k][i]['edge']
                interior_index_sub_g = self.smoother_index_edge[k][i]['interior']
                gamma_index_sub_g = np.append(face_index_sub_g, edge_index_sub_g)
                index = np.append(interior_index_sub_g, gamma_index_sub_g)
                stiff_sub = self.stiffness_matrix[k].tocsr()[index, :].tocsc()[:, index].tocsr()
                self.sc_solver[k].append(splu(stiff_sub.tocsc()))

        if self.smoothing_method == 'vertex_gamma_is':
            coarse_vertex_size = self.mesh[k - 1].get_vertex_size()
            coarse_boundary_vertex_size = len(self.mesh[k - 1].get_boundary_vertex())
            coarse_interior_vertex_size = coarse_vertex_size - coarse_boundary_vertex_size

            for i in np.arange(coarse_interior_vertex_size):
                face_index_sub_g = self.smoother_index_vertex[k][i]['face']
                edge_index_sub_g = self.smoother_index_vertex[k][i]['edge']
                interior_index_sub_g = self.smoother_index_vertex[k][i]['interior']
                gamma_index_sub_g = np.append(face_index_sub_g, edge_index_sub_g)
                index = np.append(interior_index_sub_g, gamma_index_sub_g)
                stiff_sub = self.stiffness_matrix[k].tocsr()[index, :].tocsc()[:, index].tocsr()
                self.sc_solver[k].append(splu(stiff_sub.tocsc()))

    def smoother_setting_interior(self, k):
        coarse_element_interior_edge = self.substructure[k - 1].get_coarse_element_interior_edge()
        stiff_sub = []
        for interior_index in coarse_element_interior_edge:
            stiff_sub.append(
                self.stiffness_matrix[k].tocsr()[interior_index, :].tocsc()[:, interior_index].tocsr().toarray())
        self.stiff_sub[k] = stiff_sub

    def smoother_setting_edge_ov(self, k):
        coarse_interior_edge = self.interior[k - 1]
        coarse_edge_interior_vertex = self.substructure[k - 1].get_coarse_edge_interior_vertex()
        neighboring_fine_edge = self.mesh[k].get_vertex()['edge']

        face_index = self.smoother_index_sc[k]['face']
        edge_index = self.smoother_index_sc[k]['edge']
        gamma_index = np.append(face_index, edge_index)

        local_gamma_face_index = np.arange(len(face_index))
        self.smoother_index_face_gamma_ov[k] = local_gamma_face_index

        s_gg = self.matrices[k]['s_gg']
        s_gg_f = s_gg.tocsr()[local_gamma_face_index, :].tocsc()[:, local_gamma_face_index].tocsr()
        self.s_gg_solver_f[k] = splu(s_gg_f.tocsc())

        local_gamma_edge_index = np.array([], dtype=int)

        for i in coarse_interior_edge:
            index_sub_i = np.array([], dtype=int)
            edge_index_i = neighboring_fine_edge[coarse_edge_interior_vertex[i][0]]
            for e in edge_index_i:
                index_sub_i = np.append(index_sub_i, np.where(gamma_index == e))

            local_gamma_edge_index = np.append(local_gamma_edge_index, index_sub_i)

        self.smoother_index_edge_gamma_ov[k] = local_gamma_edge_index

        s_gg_e = s_gg.tocsr()[local_gamma_edge_index, :].tocsc()[:, local_gamma_edge_index].tocsr()
        self.s_gg_solver_e[k] = splu(s_gg_e.tocsc())

    def smoother(self, x, k):
        if self.smoothing_method == 'edge_gamma_is':
            return self.smoother_edge_gamma_is(x, k)
        elif self.smoothing_method == 'vertex_gamma_is':
            return self.smoother_vertex_gamma_is(x, k)
        elif self.smoothing_method == 'face_edge_ov':
            return self.smoother_face_edge_ov(x, k)

    def smoother_vertex_gamma_is(self, x, k):
        y = np.zeros_like(x)
        interior_index = self.smoother_index_sc[k]['interior']
        face_index = self.smoother_index_sc[k]['face']
        edge_index = self.smoother_index_sc[k]['edge']
        gamma_index = np.append(face_index, edge_index)
        a_ig = self.matrices[k]['a_ig']
        a_gi = self.matrices[k]['a_gi']

        xt = self.a_ii_solver[k].solve(x[interior_index])
        xg = x[gamma_index] - a_gi * xt
        yg = np.zeros_like(xg)

        coarse_face_size = self.mesh[k - 1].get_face_size()
        coarse_boundary_face_size = len(self.mesh[k - 1].get_boundary_face())
        coarse_interior_face_size = coarse_face_size - coarse_boundary_face_size

        coarse_vertex_size = self.mesh[k - 1].get_vertex_size()
        coarse_boundary_vertex_size = len(self.mesh[k - 1].get_boundary_vertex())
        coarse_interior_vertex_size = coarse_vertex_size - coarse_boundary_vertex_size

        for i in np.arange(coarse_interior_vertex_size):
            face_index_sub = np.array([], dtype=int)
            for j in self.smoother_index_vertex[k][i]['face_index_interior']:
                face_index_sub = np.append(face_index_sub, np.arange(4 * j, 4 * (j + 1)))
            offset = 4 * coarse_interior_face_size
            edge_index_sub = np.array([], dtype=int)
            for j in self.smoother_index_vertex[k][i]['edge_index_interior']:
                edge_index_sub = np.append(edge_index_sub, np.arange(offset + 2 * j, offset + 2 * (j + 1)))
            gamma_index_sub = np.append(face_index_sub, edge_index_sub)
            size_interface = gamma_index_sub.shape[0]
            size_index = self.smoother_index_vertex[k][i]['interior'].shape[0] + size_interface
            zt = np.zeros((size_index, 1))
            zt[-size_interface:] = xg[gamma_index_sub]
            z = self.sc_solver[k][i].solve(zt)
            zg = z[-size_interface:]

            yg[gamma_index_sub] = yg[gamma_index_sub] + zg

        y[interior_index] = xt - self.a_ii_solver[k].solve(a_ig * yg)
        y[gamma_index] = y[gamma_index] + yg

        y = self.damping_factor * y
        return y

    def smoother_edge_gamma_is(self, x, k):
        y = np.zeros_like(x)
        interior_index = self.smoother_index_sc[k]['interior']
        face_index = self.smoother_index_sc[k]['face']
        edge_index = self.smoother_index_sc[k]['edge']
        gamma_index = np.append(face_index, edge_index)
        a_ig = self.matrices[k]['a_ig']
        a_gi = self.matrices[k]['a_gi']

        xt = self.a_ii_solver[k].solve(x[interior_index])
        xg = x[gamma_index] - a_gi * xt
        yg = np.zeros_like(xg)

        coarse_face_size = self.mesh[k - 1].get_face_size()
        coarse_boundary_face_size = len(self.mesh[k - 1].get_boundary_face())
        coarse_interior_face_size = coarse_face_size - coarse_boundary_face_size

        coarse_interior_edge_size = len(self.interior[k - 1])

        for i in np.arange(coarse_interior_edge_size):
            face_index_sub = np.array([], dtype=int)
            for j in self.smoother_index_edge[k][i]['face_index_interior']:
                face_index_sub = np.append(face_index_sub, np.arange(4 * j, 4 * (j + 1)))
            offset = 4 * coarse_interior_face_size
            edge_index_sub = np.array([], dtype=int)
            edge_index_sub = np.append(edge_index_sub, np.arange(offset + 2 * i, offset + 2 * (i + 1)))
            gamma_index_sub = np.append(face_index_sub, edge_index_sub)

            size_interface = gamma_index_sub.shape[0]
            size_index = self.smoother_index_edge[k][i]['interior'].shape[0] + size_interface
            zt = np.zeros((size_index, 1))
            zt[-size_interface:] = xg[gamma_index_sub]
            z = self.sc_solver[k][i].solve(zt)
            zg = z[-size_interface:]

            yg[gamma_index_sub] = yg[gamma_index_sub] + zg

        y[interior_index] = y[interior_index] + xt
        y[interior_index] = y[interior_index] - self.a_ii_solver[k].solve(a_ig * yg)
        y[gamma_index] = y[gamma_index] + yg

        y = self.damping_factor * y

        return y

    def smoother_face_edge_ov(self, x, k):
        y = np.zeros_like(x)
        interior_index = self.smoother_index_sc[k]['interior']
        face_index = self.smoother_index_sc[k]['face']
        edge_index = self.smoother_index_sc[k]['edge']
        gamma_index = np.append(face_index, edge_index)

        local_gamma_face_index = self.smoother_index_face_gamma_ov[k]
        local_gamma_edge_index = self.smoother_index_edge_gamma_ov[k]

        a_ig = self.matrices[k]['a_ig']
        a_gi = self.matrices[k]['a_gi']

        xt = self.a_ii_solver[k].solve(x[interior_index])
        xg = x[gamma_index] - a_gi * xt
        yg = np.zeros_like(xg)

        zg = self.s_gg_solver_f[k].solve(xg[local_gamma_face_index])
        yg[local_gamma_face_index] = yg[local_gamma_face_index] + zg

        zg = self.s_gg_solver_e[k].solve(xg[local_gamma_edge_index])
        yg[local_gamma_edge_index] = yg[local_gamma_edge_index] + zg

        y[interior_index] = xt - self.a_ii_solver[k].solve(a_ig * yg)
        y[gamma_index] = y[gamma_index] + yg

        y = self.damping_factor * y
        return y

    def apply_inverse_interior(self, x, k):
        y = np.zeros_like(x)
        coarse_element_interior_edge = self.substructure[k - 1].get_coarse_element_interior_edge()
        for i in np.arange(len(coarse_element_interior_edge)):
            stiff_sub = self.stiff_sub[k][i]
            interior_index = coarse_element_interior_edge[i]
            y[interior_index] = np.linalg.solve(stiff_sub, x[interior_index])
        return y

    def apply_inverse_sc(self, x, k, interior_index, interface_index):
        size_interface = interface_index.shape[0]
        index = np.append(interior_index, interface_index)
        size_index = index.shape[0]
        y = np.zeros((size_index, 1))
        y[-size_interface:] = x
        stiff_sub = self.stiffness_matrix[k].tocsr()[index, :].tocsc()[:, index].tocsr()
        z = np.linalg.solve(stiff_sub.toarray(), y)
        return z[-size_interface:]

    def mg_cycle(self, rhs, k, z0):
        print('level : ' + str(k) + ' starts')
        interior = self.interior[k]
        x = np.zeros_like(rhs)
        if k == 0:
            x[interior] = spsolve(self.stiffness_matrix_interior[k], rhs[interior]).reshape(-1, 1)
            print('level : ' + str(k) + ' ends')
            return x
        else:
            x = z0
            for _ in np.arange(self.num_presmoothing):
                r = np.zeros_like(rhs)
                r[interior] = rhs[interior] - self.stiffness_matrix_interior[k] * x[interior]
                x = x + self.smoother(r, k)

            r = np.zeros_like(rhs)
            r[interior] = rhs[interior] - self.stiffness_matrix_interior[k] * x[interior]
            rhs_coarse = self.f2c[k - 1] * r
            q = np.zeros((self.mesh[k - 1].get_edge_size(), 1))
            for _ in np.arange(self.num_coarse_grid_correction):
                q = self.mg_cycle(rhs_coarse, k - 1, q)
            x = x + self.c2f[k - 1] * q

            for _ in np.arange(self.num_postsmoothing):
                r = np.zeros_like(rhs)
                r[interior] = rhs[interior] - self.stiffness_matrix_interior[k] * x[interior]
                x = x + self.smoother(r, k)
            print('level : ' + str(k) + ' ends')
            return x
