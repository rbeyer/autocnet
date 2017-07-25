import networkx as nx
import numpy as np
import pandas as pd

"""
Which mthods must some object that handles matching expose?

-
"""

class ControlMediator(object):
    def __init__(self, candidategraph, controlnetwork):
        self._gc = candidategraph
        self._cn = controlnetwork

    @classmethod
    def from_candidategraph(cls, candidategraph, clean_keys=[]):
        mediator = cls(candidategraph, ControlNetwork())
        matches = candidategraph.get_matches(clean_keys=clean_keys)
        for match in matches:
            for idx, row in match.iterrows():
                edge = (row.source_image, row.destination_image)
                source_key = (row.source_image, row.source_idx)
                source_fields = row[['source_x', 'source_y']]
                destin_key = (row.destination_image, row.destination_idx)
                destin_fields = row[['destination_x', 'destination_y']]
                if mediator._cn.measure_to_point.get(source_key, None) is not None:
                    tempid = mediator._cn.measure_to_point[source_key]
                    mediator._cn.add_measure(destin_key, edge, row.name, source_fields, point_id=tempid)
                elif mediator._cn.measure_to_point.get(destin_key, None) is not None:
                    tempid = mediator._cn.measure_to_point[destin_key]
                    mediator._cn.add_measure(source_key, edge, row.name,  destin_fields, point_id=tempid)
                else:
                    mediator._cn.add_measure(source_key, edge, row.name,  source_fields)
                    mediator._cn.add_measure(destin_key, edge,row.name,  destin_fields)
                    mediator._cn._point_id += 1

        mediator._cn.data.index.name = 'measure_id'
        return mediator

    def identify_potential_overlaps(self, overlap=True):
        """
        Identify those points that could have additional measures
        """
        # Grab the image IDs that are in the cnet
        iids = self._cn.get_image_ids()

        # For each

    def find_candidate_measures(self, overlap=True):
        """
        Find measures that could exist in overlapping images.
        """
        pass

    def update_control_xyz(self):
        """
        """



class ControlNetwork(object):
    measures_keys = ['point_id', 'image_index', 'keypoint_index', 'edge', 'match_idx', 'x', 'y']

    def __init__(self):
        self._point_id = 0
        self._measure_id = 0
        self.measure_to_point = {}
        self.data = pd.DataFrame(columns=self.measures_keys)

    def add_from_matches(self, matches):
        for match in matches:
            for idx, row in match.iterrows():
                edge = (row.source_image, row.destination_image)
                source_key = (row.source_image, row.source_idx)
                source_fields = row[['source_x', 'source_y']]
                destin_key = (row.destination_image, row.destination_idx)
                destin_fields = row[['destination_x', 'destination_y']]
                if self.measure_to_point.get(source_key, None) is not None:
                    tempid = self.measure_to_point[source_key]
                    self.add_measure(destin_key, edge, row.name, source_fields, point_id=tempid)
                elif self.measure_to_point.get(destin_key, None) is not None:
                    tempid = self.measure_to_point[destin_key]
                    self.add_measure(source_key, edge, row.name,  destin_fields, point_id=tempid)
                else:
                    self.add_measure(source_key, edge, row.name,  source_fields)
                    self.add_measure(destin_key, edge,row.name,  destin_fields)
                    self._point_id += 1

        self.data.index.name = 'measure_id'

    def add_measure(self, key, edge, match_idx, fields, point_id=None):
        """
        Create a new measure that is coincident to a given point.  This method does not
        create the point if is missing.  When a measure is added to the graph, an associated
        row is added to the measures dataframe.

        Parameters
        ----------
        key : hashable
                  Some hashable id.  In the case of an autocnet graph object the
                  id should be in the form (image_id, match_id)

        point_id : hashable
                   The point to link the node to.  This is most likely an integer, but
                   any hashable should work.
        """
        if key in self.measure_to_point.keys():
            return
        if point_id == None:
            point_id = self._point_id
        self.measure_to_point[key] = point_id
        # The node_id is a composite key (image_id, correspondence_id), so just grab the image
        image_id = key[0]
        match_id = key[1]
        self.data.loc[self._measure_id] = [point_id, image_id, match_id, edge, match_idx, *fields]
        self._measure_id += 1

    def coverage_per_node(self):
        """
        Create a dictionary of points with each value being a boolean
        dataframe indicating whether all possible images are covered by
        an associated measure.  This supports understanding if the
        point has measures 'punched through' all images.

        """
        pass

    def get_point(self, point_id):
        """
        Get the graph and dataframe that describe a point.

        TODO: Implement the extraction of the point sub-graph from
        the parent graph.
        """
        point_graph = None
        point_frame = self.points.loc[point_id]
        return point_graph, point_frame

    def get_measure(self, measure_id):
        """
        Get an individual measure and the measure data frame.

        TODO: Implement the extraction of the measure subgraph (the linked
        point and other measures) from the parent graph.
        """
        measure_frame = self.data.loc[measure_id]
        return measure_frame

    def get_image_names(self):
        """
        Get the names of the images included in the control network.
        """
        return [cg.node[i]['image_name'] for i in self.get_image_ids()]

    def get_image_ids(self):
        """
        Get the node ids of the images in the control network.
        """
        return self.data.image_index.unique()

    def deepen_point(self, point_id):
        """
        This method seeks to deepen a given point by selecting those images that
        do not have associated measures and utilizing the CandidateGraph object to
        add add additional correspondences.
        """
        raise NotImplementedError

    def find_candidates(self, overlap=True):
        pass

    def get_measures_by_image_id(self, images):
        """
        Given a list of image names or CandidateGraph nodes identifiers, return a dataframe with the associated
        measures.
        """
        raise NotImplementedError

'''
def add_keypoints(network, kp_store):
    """
    Given a network that identifies shared correspondences between images
    and a keypoint store that has the x,y location of each keypoint,
    add an attirbute to each node in the network to store the keypoint location.

    This allows for a very lightweight correspondence network to be 'inflated'
    to contain all of the necessary correspondence information.

    Parameters
    ----------
    network : nx.Graph
              of x components, where each sub-graph is a mapping of n-correspondences
              shared across some number of images

    kp_store : CandidateGraph

    """
    df = pd.DataFrame(network.nodes(), columns=['node', 'idx']).astype(np.int)
    for i, g in df.groupby(['node']):
        node = cg.node[i]
        idx = np.unique(g)
        kps = node.get_raw_keypoint_coordinates(index=idx)
        for j, k in enumerate(g['idx']):
            x, y = kps[j]
            n.node[(i, k)]['x'] = x
            n.node[(i, k)]['y'] = y

def create_net(a):
    """
    From a pandas dataframe, create a networkx graph where the first two columns
    are a composite key for one node and the next two columns are a composite
    key for the final node.

    Parameters
    ----------
    a : dataframe
        of ndarray where rows are a to be created edge in the graph
    """
    G = nx.Graph()
    G.graph['images'] = set()
    for i in a.values:
        G.add_edge((i[0], i[1]),(i[2], i[3]), attr_dict={'x':None,
                                                         'y':None,
                                                         'subpixel':False})
        G.graph['images'].add(i[0])
        G.graph['images'].add(i[2])
    return G

def generate_control_network(graph, clean_keys=['fundamental']):
    """
    Given a graph object and a set of clean keys, extract a dataframe containing
    'source_image', 'source_idx', 'destination_image', 'destination_idx', and 'pid'.  This
    data structure represents the correspondences aggregated by an arbitrary 'pid'

    Parameters
    ----------
    graph : object
            An autocnet graph object

    clean_keys : list
                 Of clean keys

    Returns
    -------
    merged : DataFrame
             containing a grouping of the correspondences keyed by a pid
    """
    to_merge = []
    merged = pd.DataFrame(columns=['source_image', 'source_idx', 'destination_image', 'destination_idx'])
    for s, d, e in graph.edges_iter(data=True):
        matches, _ = e.clean(clean_keys=['fundamental'])
        m = matches[['source_image', 'source_idx', 'destination_image', 'destination_idx']]
        to_merge.append(m)
    merged = pd.concat(to_merge)
    G = create_net(merged)

    G.add

def merge(*networks):
    """
    Given two or more networks, merge them.

    Parameters
    ----------
    *networks : iterable
                of nx. Graph objects to merge

    Returns
    -------
    G : nx.Graph
        that is the composition of all input graphs
    """
    G = networks[0]
    for a in networks[1:]:
        if not isinstance(a, nx.Graph):
            raise TypeError
        G = nx.compose(G, a)
    return G

class Point(object):
    """
    An n-image correspondence container class to store
    information common to all identical correspondences across
    an image set.

    Attributes
    ----------
    point_id : int
               A unique identifier for the given point

    subpixel : bool
               Whether or not the point has been subpixel registered

    point_type : an ISIS identifier for the type of the point
                 as defined in the ISIS protobuf spec.

    correspondences : list
                      of image correspondences
    """
    __slots__ = '_subpixel', 'point_id', 'point_type', 'correspondences'

    def __init__(self, pid, point_type=2):
        self.point_id = pid
        self._subpixel = False
        self.point_type = point_type
        self.correspondences = []

    def __repr__(self):
        return str(self.point_id)

    def __eq__(self, other):
        return self.point_id == other

    def __hash__(self):
        return hash(self.point_id)

    @property
    def subpixel(self):
        return self._subpixel

    @subpixel.setter
    def subpixel(self, v):
        if isinstance(v, bool):
            self._subpixel = v
        if self._subpixel is True:
            self.point_type = 3


class Correspondence(object):
    """
    A single correspondence (image measure).

    Attributes
    ----------

    id : int
         The index of the point in a matches dataframe (stored as an edge attribute)

    x : float
        The x coordinate of the measure in image space

    y : float
        The y coordinate of the measure in image space

    measure_type : int
                   The ISIS measure type as per the protobuf spec

    serial : str
             A unique serial number for the image the measure corresponds to
             In the case of an ISIS cube, this is a valid ISIS serial number,
             else, None.
    """
    __slots__ = 'id', 'x', 'y', 'measure_type', 'serial'

    def __init__(self, id, x, y, measure_type=2, serial=None):
        self.id = id
        self.x = x
        self.y = y
        self.measure_type = measure_type
        self.serial = serial

    def __repr__(self):
        return str(self.id)

    def __eq__(self, other):
        return self.id == other

    def __hash__(self):
        return hash(self.id)


class CorrespondenceNetwork(object):
    """
    A container of points and associated correspondences.  The primary
    data structures are point_to_correspondence and correspondence_to_point.
    These two attributes store the mapping between point and correspondences.

    Attributes
    ----------
    point_to_correspondence : dict
                              with key equal to an instance of the Point class and
                              values equal to a list of Correspondences.

    correspondence_to_point : dict
                              with key equal to a correspondence identifier (not the class) and
                              value equal to a unique point_id (not an instance of the Point class).
                              This attribute serves as a low memory reverse lookup table

    point_id : int
               The current 'new' point id if an additional point were to be added

    n_points : int
               The number of points in the CorrespondenceNetwork

    n_measures : int
                 The number of Correspondences in the CorrespondenceNetwork

    creationdate : str
                   The date the instance of this class was first instantiated

    modifieddata : str
                   The date this class last had correspondences and/or points added
    """
    def __init__(self):
        self.point_to_correspondence = collections.defaultdict(list)
        self.correspondence_to_point = {}
        self.point_id = 0
        self.creationdate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @property
    def n_points(self):
        return len(self.point_to_correspondence.keys())

    @property
    def n_measures(self):
        return len(self.correspondence_to_point.keys())

    def add_correspondences(self, edge, matches):
        # Convert the matches dataframe to a dict
        df = matches.to_dict()
        source_image = next(iter(df['source_image'].values()))
        destination_image = next(iter(df['destination_image'].values()))

        # TODO: Handle subpixel registration here
        s_kps = edge.source.get_keypoint_coordinates().values
        d_kps = edge.destination.get_keypoint_coordinates().values

        # Load the correspondence to point data structure
        for k, source_idx in df['source_idx'].items():
            source_idx = int(source_idx)
            p = Point(self.point_id)

            destination_idx = int(df['destination_idx'][k])

            sidx = Correspondence(source_idx, *s_kps[int(source_idx)], serial=edge.source.isis_serial)
            didx = Correspondence(destination_idx, *d_kps[int(destination_idx)], serial=edge.destination.isis_serial)

            p.correspondences = [sidx, didx]

            self.correspondence_to_point[(source_image, source_idx)] = self.point_id
            self.correspondence_to_point[(destination_image, destination_idx)] = self.point_id

            self.point_to_correspondence[p].append((source_image, sidx))
            self.point_to_correspondence[p].append((destination_image, didx))

            self.point_id += 1
        self._update_modified_date()

    def _update_modified_date(self):
        self.modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def to_dataframe(self):
        pass'''
