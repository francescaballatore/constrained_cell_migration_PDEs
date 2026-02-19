# Libraries {{{
from dolfinx.io import XDMFFile, VTKFile
import os, sys
# }}}

# Class Output {{{
class Output(object):
    # Properties {{{
    @property
    def domain(self):
        return self._domain
    @property
    def functions(self):
        return self._functions
    @property
    def vtk_results(self):
        return self._vtk_results
    # @property
    # def xdmf_results(self):
    #     return self._xdmf_results
    @property
    def filename(self):
        return self._filename
    @property
    def comm(self):
        return self._comm
    # }}}
    # __init__ {{{
    def __init__(self, domain, functions, names, filename, comm, oldWriter = None):
        # Register functions and domain
        self._functions = functions
        self._domain = domain
        self._filename = filename
        # Assign names
        for k1 in range(len(names)):
            functions[k1].name = names[k1]
        # Initialisation of files
        if oldWriter is None:
            join = os.path.join
            filedir, basename = os.path.split(filename)
            filepath = "results/" + join(filedir, join("vtk", basename + ".pdv"))
            vtk_results = VTKFile(comm, filepath, "w")
            # xdmf_results = XDMFFile(comm, "results/" + filename + ".xdmf", "w")
            # xdmf_results.write_mesh(domain)
            # Register result files
            self._vtk_results = vtk_results
            # self._xdmf_results = xdmf_results
        else:
            self._vtk_results = oldWriter["vtk"]
            # self._xdmf_results = oldWriter["xdmf"]
        return
    # }}}
    # Write results {{{
    def WriteResults(self, t):
        self.vtk_results.write_function(self.functions, t)
        # for func in self.functions:
        #     self.xdmf_results.write_function(func, t)
        return
    # }}}
    # Update last mesh {{{
    def UpdateLastMesh(self):
        # self.xdmf_results.close()
        # os.system("rm results/" + self.filename + ".xdmf")
        os.system("rm results/" + self.filename + ".h5")
        # self._xdmf_results = XDMFFile(comm, "results/" + self.filename + ".xdmf", "w")
        # self.xdmf_results.write_mesh(domain)
        return
    # }}}
    # Close {{{
    def Close(self):
        # self.xdmf_results.close()
        self.vtk_results.close()
    # }}}
# }}}
