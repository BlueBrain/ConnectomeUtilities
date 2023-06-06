# SPDX-License-Identifier: Apache-2.0
import numpy


class MorphologyPathDistanceCalculator(object):
    """
    A helper to quickly calculate path distances along the morphology of a neuron. 
    
    Relies on pre-calculating a lot of data and offsets for a single morphology. This is only really
    worth the cost when you want to look up distances between hundreds of points on a single morphology.
    For smaller use cases (or cases where the distances are to be calculated on many different morphologies)
    other code may be faster. 

    NOTE: This class makes the strong assumption that the first section in the morphology is a direct descendant
    of the soma.
    """
    def __init__(self, morphology):
        """
        Args:
          morphology (morphio.Morphology): The morphology you want to calculate path distances for.
        """
        assert 0 in morphology.connectivity[-1], "This class requires the first section to be connected to the soma!"
        self.m = morphology
        self.D, self.R, self.O = self.__initialize_tables__()
    
    @staticmethod
    def segment_offset_table(m):
        """
        Calculate for each segment in each section the distance from the root of the section to the 
        root of the segment. Since different sections have different numbers of segments, all lists
        of offsets are padded with NaN and stacked to a 2d array.

        Args:
          m (morphio.Morphology): The morphogy in question
        
        Returns:
          offset_table (numpy.array): Table of segment offsets. To be indexed as offset_table[sec_id][seg_id].
          Note: sec_id is base-0, i.e. index 0 is NOT the soma. The soma is not listed at all (assumed to be just a point).
        """
        def cumulative_section_length(sec):
            seg_lengths = numpy.sqrt(numpy.sum(numpy.diff(sec.points, axis=0) ** 2, axis=1))
            return [0] + list(numpy.cumsum(seg_lengths))
    
        offset_lists = [cumulative_section_length(_sec)
                        for _sec in m.sections]
        L = numpy.max([len(_x) for _x in offset_lists])
        offset_lists = [_x + [numpy.NaN for _ in range(L - len(_x))]
                       for _x in offset_lists]
        return numpy.vstack(offset_lists)

    def __initialize_tables__(self):
        """
        Calculate the path distances between the roots of pairs of sections, and their relation within the morphology tree.

        Returns:
            root_D_tbl (numpy.array): The table of pairwise path distances between the roots of all sections of the morphology.
            relation_tbl (numpy.array): The table of tree-relations of all pairs of sections. Entries are integers with the following
            meaning:
              -1: non-initialized; this should never happen
              0: Same section; i.e. this should only occur along the main diagonal
              1: tbl[i, j] == 1 means section j is the ancestor of section i
              2: tbl[i, j] == 2 means section j is the descendant of section i
              3: tbl[i, j] == 3 means section i and j are siblings
            O (numpy.array): See segment_offset_table
        """
        m = self.m
        O = self.segment_offset_table(m)
        L = numpy.nanmax(O, axis=1)

        root_D_tbl = -numpy.ones((len(m.sections), len(m.sections)))
        root_D_tbl[numpy.diag_indices_from(root_D_tbl)] = 0.0
        relation_tbl = -numpy.ones((len(m.sections), len(m.sections)), dtype=int) # -1 non-init; 0 same; 1 j ancestor; 2 j descendant; 3 siblings
        relation_tbl[numpy.diag_indices_from(relation_tbl)] = 0

        def recursive_fill(called_sec, parent, with_length):
            called = called_sec.id
            if parent >= 0:
                root_D_tbl[called, parent] = with_length
                root_D_tbl[parent, called] = with_length
                relation_tbl[called, parent] = 1
                relation_tbl[parent, called] = 2

                for j, rows in enumerate(zip(root_D_tbl[parent], relation_tbl[parent])):
                    row_D, row_rel = rows
                    if j == called: continue
                    if row_rel == 1: # generalized node is ancestor of parent node
                        relation_tbl[called, j] = 1
                        root_D_tbl[called, j] = row_D + with_length
                        relation_tbl[j, called] = 2
                        root_D_tbl[j, called] = row_D + with_length
                    elif row_rel == 2: # generalized node is descendant of parent node
                        relation_tbl[called, j] = 3
                        root_D_tbl[called, j] = row_D - with_length
                        relation_tbl[j, called] = 3
                        root_D_tbl[j, called] = row_D - with_length
                    elif row_rel == 3: # generalized node is sibling of parent node
                        relation_tbl[called, j] = 3
                        root_D_tbl[called, j] = row_D + with_length
                        relation_tbl[j, called] = 3
                        root_D_tbl[j, called] = row_D + with_length
            for child in called_sec.children:
                recursive_fill(child, called, L[called])

        for child1 in m.root_sections:
            cid1 = child1.id
            for child2 in m.root_sections:
                cid2 = child2.id
                if cid1 != cid2:
                    root_D_tbl[cid1, cid2] = 0.0
                    relation_tbl[cid1, cid2] = 3
        for child in m.root_sections:
            recursive_fill(child, -1, 0)
        
        return root_D_tbl, relation_tbl, O
    
    def within_section_offsets(self, locs,
                               str_section_id="afferent_section_id",
                               str_segment_id="afferent_segment_id",
                               str_offset="afferent_segment_offset"):
        """
        Calculate the offset of points within a section.

        Args:
          locs (pandas.DataFrame): dataframe where columns specify the section id, segment id and within-segment offset
          of points on the neuron morphology. 
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.
        
        Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
        Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma, index 1 is the first "proper" section.
        As path distance along the soma is a tricky concept, the function will return 0.0 for all entries on the soma!

        Returns:
          offsets (numpy.array): The within-section offsets in the same order as the input.
        """
        offsets = self.O[locs[str_section_id].values - 1, locs[str_segment_id]]
        assert not numpy.any(numpy.isnan(offsets))
        offsets = offsets + locs[str_offset].values
        offsets[locs[str_section_id].values == 0] = 0.0
        return offsets
    
    def path_distances(self, locs_from, locs_to=None,
                       str_section_id="afferent_section_id",
                       str_segment_id="afferent_segment_id",
                       str_offset="afferent_segment_offset",
                       same_section_only=False):
        """
        Calculate path distances. Either pairwise, or between two sets of points.
        Args:
          locs_from (pandas.DataFrame): Locations of points on the morphology represented by this object
          to calculate the distances from. A dataframe where columns specify the section id, segment id and within-segment offset
          of points on the neuron morphology.
          locs_to (pandas.DataFrame, default=None): Locations to calculate the distances to. If none, locs_from is used.
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.
          same_section_only (bool, default=False): If True, set values for all pairs NOT on the same section to NaN.

          Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
          Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma, index 1 is the first "proper" section.
          This is also compatible with how bluepysnap returns the information.
          Since the concept of path distance along the soma is tricky, this method assumes that the soma is a point.
          Consequently, for all locations on the soma, the values will be as if the location was at the root of a root section.
        
        Returns:
          dist (numpy.array): Array of pairwise path distances.
        """
        if locs_to is None:
            locs_to = locs_from
        
        o_from = self.within_section_offsets(locs_from, str_section_id=str_section_id,
                                             str_segment_id=str_segment_id, str_offset=str_offset)
        o_to = self.within_section_offsets(locs_to, str_section_id=str_section_id,
                                           str_segment_id=str_segment_id, str_offset=str_offset)
        if same_section_only:
            same_section = numpy.eye(len(self.m.sections) + 1, dtype=bool)
            same_section = same_section[numpy.ix_(locs_from[str_section_id], locs_to[str_section_id])]
            dist = numpy.NaN * numpy.ones(same_section.shape, dtype=float)
            i, j = numpy.nonzero(same_section)
            dist[i, j] = numpy.abs(o_from[i] - o_to[j])
            return dist
        
        sec_idx_fr = locs_from[str_section_id].copy() - 1
        sec_idx_to = locs_to[str_section_id].copy() - 1
        # Since we test above that the first section is rooted at the soma, we can use that section in place of the soma.
        # This makes sense, because we assume the soma to be a point (and consequently, within_section_offset) returns
        # 0.0 for all locations on the soma.
        sec_idx_fr[sec_idx_fr < 0] = 0; sec_idx_to[sec_idx_to < 0] = 0

        dist = self.D[numpy.ix_(sec_idx_fr,
                                sec_idx_to)].copy()
        rel = self.R[numpy.ix_(sec_idx_fr,
                                sec_idx_to)]
        
        i, j = numpy.nonzero(rel == 0) # on same section
        dist[i, j] = dist[i, j] + numpy.abs(o_from[i] - o_to[j])

        i, j = numpy.nonzero(rel == 1) # locs_to (j) is ancestor
        dist[i, j] = dist[i, j] + o_from[i] - o_to[j]
        
        i, j = numpy.nonzero(rel == 2) # locs_from (i) is ancestor
        dist[i, j] = dist[i, j] - o_from[i] + o_to[j]
        
        i, j = numpy.nonzero(rel == 3) # siblings
        dist[i, j] = dist[i, j] + o_from[i] + o_to[j]
        return dist
