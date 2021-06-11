from os.path import expanduser, join, dirname

import numpy as np
import pickle
import mfem.ser as mfem


class UniformMesh:
    def __init__(self, n, mesh_name):
        self.__n = n
        self.__mesh_name = mesh_name

        self.__initialized = False

        self.__mesh = None #
        self.__mesh_file = expanduser(join(dirname(__file__), 'data', mesh_name + '.mesh')) #

        self.__coordinate = None
        self.__vertex = dict()
        self.__edge = dict()
        self.__face = dict()
        self.__element = dict()

        self.__boundary_vertex = np.array([], dtype=int)
        self.__boundary_edge = np.array([], dtype=int)
        self.__boundary_face = np.array([], dtype=int)

        self.__file_prefix = ''
        if self.__mesh_name != '':
            self.__file_prefix = self.__mesh_name + '_'

        self.initialize()

    def initialize(self):
        print('Initialize : Mesh {}'.format(self.__n))
        self.__mesh = mfem.Mesh(self.__mesh_file, 1, 1)
        ref_levels = int(np.log(self.__n) / np.log(2.0)) - 1
        for _ in np.arange(ref_levels):
            self.__mesh.UniformRefinement()
        self.__mesh.PrintInfo()
        mesh_filename = self.__file_prefix + 'mesh' + str(self.__n) + '.pickle'
        mesh = pickle.load(open(mesh_filename, 'rb'))
        self.__coordinate = mesh['coordinate']
        self.__vertex = mesh['vertex']
        self.__edge = mesh['edge']
        self.__face = mesh['face']
        self.__element = mesh['element']

        boundary_filename = self.__file_prefix + 'boundary' + str(self.__n) + '.pickle'
        boundary = pickle.load(open(boundary_filename, 'rb'))
        self.__boundary_vertex = boundary['vertex']
        self.__boundary_edge = boundary['edge']
        self.__boundary_face = boundary['face']

        self.__initialized = True

    def is_initialized(self):
        return self.__initialized

    def get_n(self):
        return self.__n

    def get_coordinate(self):
        return self.__coordinate

    def get_vertex(self):
        return self.__vertex

    def get_element(self):
        return self.__element

    def get_edge(self):
        return self.__edge

    def get_face(self):
        return self.__face

    def get_vertex_dof(self):
        return self.__element['vertex']

    def get_edge_dof(self):
        return self.__element['edge']

    def get_face_dof(self):
        return self.__element['face']

    def get_boundary_vertex(self):
        return self.__boundary_vertex

    def get_boundary_edge(self):
        return self.__boundary_edge

    def get_boundary_face(self):
        return self.__boundary_face

    def get_mesh_name(self):
        return self.__mesh_name

    def get_number_attribute(self):
        return np.amax(self.__element['attribute'])

    def get_attribute(self):
        return self.__element['attribute']

    def get_vertex_size(self):
        return self.__mesh.GetNV()

    def get_edge_size(self):
        return self.__mesh.GetNEdges()

    def get_face_size(self):
        return self.__mesh.GetNFaces()

    def get_element_size(self):
        return self.__mesh.GetNE()
