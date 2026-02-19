# Libraries {{{
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import gmsh
from pdb import set_trace

from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
# }}}

# Class to make and modify mesh {{{
class DynamicMesh(object):
    # Properties {{{
    @property
    def lc(self):
        return self._lc
    @property
    def splines(self):
        return self._splines
    @property
    def model(self):
        return self._model
    @property
    def meshOrder(self):
        return self._meshOrder
    @property
    def CreateGeometryFromLines(self):
        return self._CreateGeometryFromLines
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._lc = kwargs["lc"]
        self._meshOrder = kwargs.get("meshOrder", 2)
        def CreateGeo():
            raise NotImplementedError("CreateGeometryFromLines has not been defined yet. It has to be defined in MakeMesh.")
        self._CreateGeometryFromLines = CreateGeo
        self._splines = None
        # Make mesh
        self.MakeMesh(**kwargs)
        return
    # }}}
    # Make mesh {{{
    def MakeMesh(self, **kwargs):
        raise NotImplementedError("Subclasses must implement MakeMesh")
    # }}}
    # Remesh {{{
    def Remesh(self, coordinates):
        """
        Keep in mind that nodes in gmsh start with the id 0.
        """
        lc = self.lc
        splines = self.splines
        meshOrder = self.meshOrder
        # Initialisation
        gmsh.clear()
        gmsh.initialize()
        model = gmsh.model
        geo = model.occ
        model.add("remesh")
        self._model = model
        # Create points
        for pCoords in coordinates:
            pid = geo.addPoint(*pCoords, lc)
        # Create splines
        loopSplines = []
        for spline in splines:
            loopSplines.append(geo.addBSpline(spline))
        # Create geometry from lines
        self._splines = self.CreateGeometryFromLines()
        # gmsh.fltk.run()
        return
    # }}}
# }}}

# Functions {{{
# Make circle mesh {{{
def MakeCircle(radius : float, lc : float, meshOrder = 1):
    # Initialisation
    gmsh.initialize()
    model = gmsh.model
    geo = model.occ
    model.add("cell")
    # Create points
    cc = geo.addPoint(0.0, 0.0, 0.0, lc)
    rp = geo.addPoint(radius, 0.0, 0.0, lc)
    lp = geo.addPoint(-radius, 0.0, 0.0, lc)
    # Create lines
    topLine = geo.addCircleArc(rp, cc, lp)
    botLine = geo.addCircleArc(lp, cc, rp)
    # Create loop
    loop = geo.addCurveLoop([topLine, botLine])
    # Synchronise
    geo.synchronize()
    # Groups
    gr_gamma = model.addPhysicalGroup(1, [topLine, botLine])
    # Mesh
    model.mesh.generate(dim = 1)
    model.mesh.setOrder(meshOrder)
    # gmsh.fltk.run()
    return model
# }}}
# Make list of ordered nodes {{{
def MakeListOfOrderedNodes(model, groupTags):
    # Get list of entities {{{
    entities = []
    for (dim, tag) in groupTags:
        if dim != 1:
            raise("Error: only edges can be considered.")
        entities.append(model.getEntitiesForPhysicalGroup(1, tag))
    entities = np.hstack(entities)
    entities = np.unique(entities)
    # }}}
    # Get list of edges {{{
    edges = []
    for entityTag in entities:
        _, eleTags, _ = model.mesh.getElements(1, entityTag)
        edges.append(eleTags)
    edges = np.hstack(edges)
    edges = np.unique(edges)
    # }}}
    # Get list of nodes {{{
    nodes = []
    for edge in edges:
        eType, nodeTags, _, _ = model.mesh.getElement(edge)
        if eType == 1:
            nodes.append(np.append(nodeTags, [-1]))
        elif eType == 8:
            nodes.append(nodeTags)
        else:
            raise TypeError("Invalid element type for edges.")
    nodes = np.array(nodes).astype(int)
    # }}}
    # Find limit nodes
    startNode, endNode = CurveLimitNodes(nodes)
    # Create ordered node list
    numEles = edges.shape[0]
    nodeList = OrderNodeList(startNode, endNode, nodes, numEles)
    return nodeList
# }}}
# Find curve limit nodes {{{
def CurveLimitNodes(nodes):
    # Get the number of appearance of each extreme node
    numAppearance = []
    for eleNodes in nodes:
        for node in eleNodes:
            numAppearance.append([node, np.count_nonzero(nodes[:, :2] == node)])
    numAppearance = np.array(numAppearance)
    numberOfLimitNodes = np.count_nonzero(numAppearance[:, 1] == 1)
    # Only two limit nodes should be present (nodes appearing only once)
    if numberOfLimitNodes != 2:
        message = "The number of limit nodes must to be 2 for open curves."
        raise ValueError(message)
    # Number of nodes appearing in more than two elements
    numberOfWrongNodes = np.count_nonzero(numAppearance[:, 1] > 2)
    if numberOfWrongNodes != 0:
        message = "There are nodes appearing in more than two elements."
        raise ValueError(message)
    # Set start and end point
    startNodeId, endNodeId = np.argwhere(numAppearance[:, 1]  == 1)
    startNode = numAppearance[startNodeId, 0][0]
    endNode = numAppearance[endNodeId, 0][0]
    return startNode, endNode
# }}}
# Create ordered node list {{{
def OrderNodeList(startNode, endNode, nodes, numEles):
    # Get the total number of elements
    # Initialisation of node list
    nodeList = [startNode]
    oldNode = startNode
    # Find the element containing the old node
    newEle = np.argwhere(nodes == oldNode)[0, 0]
    # Get the nodes of the element (start, end, middle)
    ns, ne, nm = nodes[newEle, :]
    # If the middle node is valid add it to the node list
    if nm != -1:
        nodeList.append(nm)
    # Find which node of the new element is the new node
    if oldNode == ns:
        newNode = ne
    elif oldNode == ne:
        newNode = ns
    else:
        raise ValueError("Invalid nodes")
    # Add the new node to the node list
    nodeList.append(newNode)
    # Repeat the process adding nodes following their connection on edges
    for k1 in range(numEles - 1):
        # Update old values
        oldNode = newNode
        oldEle = newEle
        # Find new element
        ele1, ele2 = np.argwhere(nodes == oldNode)[:, 0]
        if oldEle == ele1:
            newEle = ele2
        elif oldEle == ele2:
            newEle = ele1
        else:
            raise ValueError("Invalid elements")
        # Get the nodes of the element (start, end, middle)
        ns, ne, nm = nodes[newEle, :]
        # If the middle node is valid add it to the node list
        if nm != -1:
            nodeList.append(nm)
        # Find which node of the new element is the new node
        if oldNode == ns:
            newNode = ne
        elif oldNode == ne:
            newNode = ns
        else:
            raise ValueError("Invalid nodes")
        nodeList.append(newNode)
    nodeList = np.array(nodeList)
    return nodeList
# }}}
# Order splines {{{
def OrderSplines(splineList):
    """
    Order splines so the first node coincides with the last node of the previous
    """
    # Get the order of the first two splines {{{
    # Get the previous validated spline end points
    preSpline = splineList[0]
    pre0 = preSpline[0]
    pref = preSpline[-1]
    # Get the current not validated spline end points
    curSpline = splineList[1]
    cur0 = curSpline[0]
    curf = curSpline[-1]
    # Choose the right order
    if pre0 == cur0 or pre0 == curf:
        newList = [np.flip(preSpline)]
    elif pref == cur0 or pref == curf:
        newList = [preSpline]
    else:
        message = "The first two splines are not connected."
        raise ValueError(message)
    # }}}
    # Continue adding splines {{{
    numSplines = len(splineList)
    for k1 in range(1, numSplines):
        # Get the previous validated spline end points
        preSpline = newList[k1 - 1]
        pref = preSpline[-1]
        # Get the current not validated spline end points
        curSpline = splineList[k1]
        cur0 = curSpline[0]
        curf = curSpline[-1]
        # Correction of order if necessary
        if pref == cur0:
            newList.append(curSpline)
        elif pref == curf:
            newList.append(np.flip(curSpline))
        else:
            print("Wrong splines")
            print(preSpline)
            print(curSpline)
            message = f"The {k1 - 1} and {k1} splines are not connected."
            raise ValueError(message)
    # }}}
    return newList
# }}}
# Order coordinates {{{
def OrderCoordinates(domain):
    # Get comm
    comm = MPI.COMM_WORLD
    # Get geometry properties
    indices = domain.geometry.input_global_indices
    coords = domain.geometry.x
    # Get global number of nodes
    globalNumNodes = comm.allreduce(np.max(indices), op = MPI.MAX) + 1
    # Initialise global ordered coordinates
    orderedCoordinates = np.zeros((globalNumNodes, coords.shape[1]))
    # Fill with current indices
    for k1 in range(len(indices)):
        inputIndex = indices[k1]
        node = coords[k1]
        orderedCoordinates[inputIndex] = node
    # Global coordinates
    globalCoordinates = np.empty((globalNumNodes, coords.shape[1]))
    recvbuf = comm.allgather(orderedCoordinates)
    recvindices = comm.allgather(indices)
    for k1 in range(len(recvbuf)):
        data = recvbuf[k1]
        indices = recvindices[k1]
        for k2 in range(len(indices)):
            inputIndex = indices[k2]
            globalCoordinates[inputIndex] = data[inputIndex]
    return globalCoordinates
# }}}
# Mesh quality {{{
def MeshQuality(domain, measure = "aspect_ratio"):
    dim = domain.topology.dim
    domain.topology.create_connectivity(1, dim)
    if measure == "aspect_ratio":
        # Create connectivity: how cells are related to edges
        domain.topology.create_connectivity(dim, 1)
        conns = domain.topology.connectivity(dim, 1)
        numCells = conns.num_nodes
        # Measure quality
        meshQua = np.zeros(numCells)
        for k1 in range(numCells):
            edges = conns.links(k1)
            lengths = domain.h(1, edges)
            meshQua[k1] = max(lengths)/min(lengths)
    else:
        message = "Unavailable mesh quality measure."
    return meshQua
# }}}
# Mesh distance to point fields {{{
def MeshDistanceToPointsField(model, points, eval_func, thresholds):
    """
    lcMin, lcMax, distMin, distMax = threshold[i]
    """
    # Make distance base field
    F0 = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(F0, "PointsList", points)
    # Make eval field over which the threshold fields will be computed
    Feval = model.mesh.field.add("MathEval")
    model.mesh.field.setString(Feval, "F", eval_func(F0))
    # Create threshold fields
    Fthrs = []
    for threshold in thresholds:
        lcMin, lcMax, distMin, distMax = threshold
        Fi = model.mesh.field.add("Threshold")
        model.mesh.field.setNumber(Fi, "InField", Feval)
        model.mesh.field.setNumber(Fi, "SizeMin", lcMin)
        model.mesh.field.setNumber(Fi, "SizeMax", lcMax)
        model.mesh.field.setNumber(Fi, "DistMin", distMin)
        model.mesh.field.setNumber(Fi, "DistMax", distMax)
        Fthrs.append(Fi)
    # Create minimum size field from threshold fields
    Fmin = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(Fmin, "FieldsList", Fthrs)
    return F0, Feval, Fthrs, Fmin
# }}}
# Mesh exponential distribution {{{
def MeshExponentialDistribution(model, listEntities, minSize, maxSize, length,
                            sampling = 1000):
    """
    Distribution:
        size = exp(a*x)*b
        - x: distance to listEntities
        - a: (1.0/length)*ln(maxSize/minSize)
        - b: minSize
    listEntities: ["typeList", [tags]]
    """
    # Make distance base field
    F0 = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(F0, listEntities[0], listEntities[1])
    model.mesh.field.setNumber(F0, "Sampling", sampling)
    # Compute distribution parameters
    a = (1.0/length)*np.log(maxSize/minSize)
    b = minSize
    eval_func = f"exp({a}*F{F0})*{b}"
    # Compute eval
    Feval = model.mesh.field.add("MathEval")
    model.mesh.field.setString(Feval, "F", eval_func)
    return F0, Feval
# }}}
# Mesh Sigmod distribution {{{
def MeshSigmoidDistribution(model, listEntities, minSize, maxSize, l_t,
                            sampling = 1000):
    """
    Distribution:
        size = b/(1.0 + exp(-a*(x - c)))
        - x: distance to listEntities
        - a: ln((maxSize/minSize) - 1)/l_t
        - b: maxSize
        - c: l_t
    listEntities: ["typeList", [tags]]
    """
    # Make distance base field
    F0 = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(F0, listEntities[0], listEntities[1])
    model.mesh.field.setNumber(F0, "Sampling", sampling)
    # Compute distribution parameters
    a = np.log((maxSize/minSize) - 1)/l_t
    b = maxSize
    c = l_t
    eval_func = f"{b}/(1.0 + exp(-{a}*(F{F0} - {c})))"
    # Compute eval
    Feval = model.mesh.field.add("MathEval")
    model.mesh.field.setString(Feval, "F", eval_func)
    return F0, Feval
# }}}
# Equidistribute mesh {{{
def EquidistributeMesh(coords, bc_type = "periodic", inSamples = 100, optimal = True):
    # Cumulative length
    spl_x, spl_y, cumSpline_length = ArcLengthSpline(coords, bc_type = bc_type,
                                                     inSamples = inSamples)
    spline_length = cumSpline_length[-1]
    numPoints = len(cumSpline_length) - 1 # No repeated points
    para = np.arange(numPoints + 1) # Parameter
    # Optimal lengths
    dumLengths = np.linspace(0.0, spline_length, numPoints + 1)
    if optimal:
        disLengths = (dumLengths - cumSpline_length)[:-1]
        delLength = -np.sum(disLengths)/numPoints
        newLengths = dumLengths - delLength
    else:
        newLengths = dumLengths
    # New positions
    interp_func = interp1d(cumSpline_length, para, kind = 'linear',
                           fill_value = "extrapolate")
    newPara = interp_func(newLengths)
    new_x = spl_x(newPara)
    new_y = spl_y(newPara)
    # New coords
    equiCoords = np.column_stack([new_x, new_y])
    return equiCoords
# }}}
# Compute arc length spline {{{
def ArcLengthSpline(coords, bc_type = "periodic", inSamples = 1000):
    # Get data to define parameterised spline
    x = coords[:, 0]
    y = coords[:, 1]
    numPoints = len(coords) - 1 # No repeated points
    para = np.arange(numPoints + 1) # Parameter
    # Define spline
    spl_x = make_interp_spline(para, x, bc_type = bc_type)
    spl_y = make_interp_spline(para, y, bc_type = bc_type)
    # Segment length
    para_segments = np.linspace(0, numPoints, inSamples*numPoints)
    dpara = (numPoints)/(inSamples*numPoints)
    dx = spl_x(para_segments, 1)
    dy = spl_y(para_segments, 1)
    step_length_point = np.sqrt(dx**2.0 + dy**2.0)*dpara
    step_length_point_half_left = step_length_point/2.0
    step_length_point_half_right = np.concatenate([step_length_point_half_left[1:],
                                                   np.array([step_length_point_half_left[0]])])
    step_length = step_length_point_half_left + step_length_point_half_right
    cumSpline_length_fine = np.cumsum(step_length)
    cumSpline_length = np.concatenate([np.zeros(1),
                                       cumSpline_length_fine[(para*inSamples - 1)[1:]]])
    return spl_x, spl_y, cumSpline_length

# }}}
# Curve interpolation {{{
def CurveInterpolation(xIn, uIn, xOut, bc_type = "periodic", inSamples = 100):
    # Get splines
    spl_x_in, spl_y_in, arcLength_in = ArcLengthSpline(xIn, bc_type = bc_type,
                                                       inSamples = inSamples)
    spl_x_out, spl_y_out, arcLength_out = ArcLengthSpline(xOut, bc_type = bc_type,
                                                           inSamples = inSamples)
    # Interpolate based on the arc length
    uOut = np.interp(arcLength_out[:-1], arcLength_in, uIn)
    return uOut
# }}}
# }}}
