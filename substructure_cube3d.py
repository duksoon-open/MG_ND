import pickle


class SubstructureCube3D:
    def __init__(self, mesh_coarse, mesh_fine):
        self.__mesh_coarse = mesh_coarse
        self.__mesh_fine = mesh_fine

        if not self.__mesh_coarse.is_initialized():
            self.__mesh_coarse.initialize()

        if not self.__mesh_fine.is_initialized():
            self.__mesh_fine.initialize()

        self.__initialized = False

        self.__file_prefix = ''
        if self.__mesh_coarse.get_mesh_name() != '':
            self.__file_prefix = self.__mesh_coarse.get_mesh_name() + '_'

        self.__n_coarse = self.__mesh_coarse.get_n()
        self.__n_fine = self.__mesh_fine.get_n()

        self.__coarse_to_fine = dict()
        self.__coarse_interior = dict()

        self.initialize()

    def initialize(self):
        print('Initialize : Substructure {}, {}'.format(self.__n_coarse, self.__n_fine))
        c2f_filename = self.__file_prefix + 'coarse_to_fine' + str(self.__n_coarse) + '_' + str(
            self.__n_fine) + '.pickle'
        self.__coarse_to_fine = pickle.load(open(c2f_filename, 'rb'))

        coarse_interior_filename = self.__file_prefix + 'coarse_interior' + str(self.__n_coarse) + '_' + str(
            self.__n_fine) + '.pickle'
        self.__coarse_interior = pickle.load(open(coarse_interior_filename, 'rb'))

        self.__initialized = True

    def is_initialized(self):
        return self.__initialized

    def get_coarse_mesh(self):
        return self.__mesh_coarse

    def get_fine_mesh(self):
        return self.__mesh_fine

    def get_coarse_to_fine_edge(self):
        return self.__coarse_to_fine['edge']

    def get_coarse_to_fine_face(self):
        return self.__coarse_to_fine['face']

    def get_coarse_to_fine_element(self):
        return self.__coarse_to_fine['element']

    def get_coarse_element_interior_vertex(self):
        return self.__coarse_interior['element']['vertex']

    def get_coarse_element_interior_edge(self):
        return self.__coarse_interior['element']['edge']

    def get_coarse_element_interior_face(self):
        return self.__coarse_interior['element']['face']

    def get_coarse_face_interior_vertex(self):
        return self.__coarse_interior['face']['vertex']

    def get_coarse_face_interior_edge(self):
        return self.__coarse_interior['face']['edge']

    def get_coarse_edge_interior_vertex(self):
        return self.__coarse_interior['edge']['vertex']
