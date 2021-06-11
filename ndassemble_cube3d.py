import numpy as np

from scipy import sparse


class NDAssembleCube3D:
	def __init__(self, mesh, alpha=1.0, beta=1.0):
		self.__mesh = mesh

		self.__file_prefix = 'ND_Assemble_'
		self.__file_suffix = '_' + str(alpha) + '_' + str(beta)
		if self.__mesh.get_mesh_name() != '':
			self.__file_prefix = self.__mesh.get_mesh_name() + '_' + self.__file_prefix

	def get_mesh(self):
		return self.__mesh

	def mass_assemble(self):
		filename = self.__file_prefix + 'Mass_' + str(self.__mesh.get_n()) + self.__file_suffix + '.npz'
		return sparse.load_npz(filename)

	def curlcurl_assemble(self):
		filename = self.__file_prefix + 'CurlCurl_' + str(self.__mesh.get_n()) + self.__file_suffix + '.npz'
		return sparse.load_npz(filename)

	def rhs_assemble(self, ftn):
		filename = self.__file_prefix + 'RHS_' + ftn.__name__ + '_' + str(
			self.__mesh.get_n()) + self.__file_suffix + '.npy'
		return np.load(filename)
