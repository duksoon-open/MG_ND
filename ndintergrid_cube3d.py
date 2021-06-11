from scipy import sparse
from scipy.sparse import lil_matrix

from substructure_cube3d import SubstructureCube3D


class NDIntergridCube3D:
    def __init__(self, mesh_coarse, mesh_fine):
        self.__mesh_coarse = mesh_coarse
        self.__mesh_fine = mesh_fine

        if not self.__mesh_coarse.is_initialized():
            self.__mesh_coarse.initialize()

        if not self.__mesh_fine.is_initialized():
            self.__mesh_fine.initialize()

        self.__initialized = False

        self.__file_prefix = 'intergrid_transfer'
        if self.__mesh_coarse.get_mesh_name() != '':
            self.__file_prefix = self.__mesh_coarse.get_mesh_name() + '_' + self.__file_prefix

        self.__n_coarse = self.__mesh_coarse.get_n()
        self.__n_fine = self.__mesh_fine.get_n()

        self.__substructure = SubstructureCube3D(self.__mesh_coarse, self.__mesh_fine)

        self.__size_edge_coarse = self.__mesh_coarse.get_edge_size()
        self.__size_edge_fine = self.__mesh_fine.get_edge_size()

        self.__nd_intergrid_transfer_f2c = lil_matrix((self.__size_edge_coarse, self.__size_edge_fine))
        self.__nd_intergrid_transfer_c2f = lil_matrix((self.__size_edge_fine, self.__size_edge_coarse))

        self.initialize()

    def initialize(self):
        print('Initialize : Intergrid Transfer {}, {}'.format(self.__n_coarse, self.__n_fine))
        intergrid_transfer_filename = self.__file_prefix + str(self.__n_coarse) + '_' + str(self.__n_fine) + '.npz'

        self.__nd_intergrid_transfer_f2c = sparse.load_npz(intergrid_transfer_filename)
        self.__nd_intergrid_transfer_c2f = self.__nd_intergrid_transfer_f2c.tocsc().T.tocsr()

        self.__initialized = True

    def is_initialized(self):
        return self.__initialized

    def get_coarse_mesh(self):
        return self.__mesh_coarse

    def get_coarse_edge_size(self):
        return self.__size_edge_coarse

    def get_fine_mesh(self):
        return self.__mesh_fine

    def get_fine_edge_size(self):
        return self.__size_edge_fine

    def get_nd_intergrid_transfer_c2f(self):
        return self.__nd_intergrid_transfer_c2f

    def get_nd_intergrid_transfer_f2c(self):
        return self.__nd_intergrid_transfer_f2c

    def get_substructure(self):
        return self.__substructure
