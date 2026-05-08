#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from pathlib import PurePath, Path
from astropy.table import Table, unique
import astropy.units as u
from numba import njit

class CTMergerTreeError(Exception):
    pass
class CTNoProgenitorError(CTMergerTreeError):
    pass
class NoDescendantError(CTMergerTreeError):
    pass
class NoFirstProgenitorError(CTMergerTreeError):
    pass
class OutOfRangeLoopError(CTMergerTreeError):
    pass
class DescendantSnapNrError(CTMergerTreeError):
    pass

force_multiple=True
force_multiple=False

@njit
def tmpfunc(tree_TreeDescendant, tree_SnapNum, tree_SubhaloNr, snapNr, SubhaloNr, targetSnapNr):
    # a large positive integer number, to avoid an infinite loop below
    Nlarge = 10_000
    out = np.zeros_like(SubhaloNr, dtype=np.bool_)

    for k in range(SubhaloNr.size):
        # using int() ensures we have an array of only one element
        ind = np.flatnonzero(
                (tree_SnapNum == snapNr) & (tree_SubhaloNr == SubhaloNr[k]))
        assert ind.size == 1
        ind = ind[0]


        for j in range(Nlarge):

            ind2 = tree_TreeDescendant[ind]

            # invalid index
            if ind2 < 0:
                # it's false by default
                # out[k] = False
                break

            if targetSnapNr == tree_SnapNum[ind2]:
                out[k] = True
                break

            ind = ind2

    return out

def read_treeTable(fname, Nfiles):

    treeT = {}


    if Nfiles > 1 or force_multiple:
        # still have to adapt this from gad4
        raise NotImplementedError

        # allocate arrays
        with h5py.File(fname.with_suffix('.0.hdf5'), 'r') as tr:
            treeT['z'] = tr['TreeTimes/Redshift'][()]
            treeT['Header'] = {key:int(val) for key, val in tr['Header'].attrs.items()}

            treeT['Length'] = np.zeros(treeT['Header']['Ntrees_Total'],
                                       dtype = tr['/TreeTable/Length'].dtype)
            treeT['StartOffset'] = np.zeros(treeT['Header']['Ntrees_Total'],
                                            dtype = tr['/TreeTable/StartOffset'].dtype)
            treeT['TreeID'] = np.zeros(treeT['Header']['Ntrees_Total'],
                                       tr['TreeTable/TreeID'].dtype)

        off = 0
        for i in range(Nfiles):
            with h5py.File(fname.with_suffix(f'.{i}.hdf5'), 'r') as tr:
                size = tr['Header'].attrs['Ntrees_ThisFile']

                if size > 0:
                    tr['/TreeTable/Length'].read_direct(treeT['Length'],
                                            dest_sel = np.s_[off:off+size])
                    tr['/TreeTable/StartOffset'].read_direct(treeT['StartOffset'],
                                            dest_sel = np.s_[off:off+size])
                    tr['TreeTable/TreeID'].read_direct(treeT['TreeID'],
                                            dest_sel = np.s_[off:off+size])

                off+= size
    else:
        with h5py.File(fname, 'r') as tr:
            # treeT['z'] = tr['TreeTimes/Redshift'][()]

            group = tr['/ForestInfo']
            treeT['Length'] = group['ForestNhalos'][()]
            treeT['StartOffset'] = group['ForestHalosOffset'][()]
            treeT['TreeID'] = group['ForestID'][()]

            # group = tr['/TreeInfo']
            # treeT['Length'] = group['TreeNhalos'][()]
            # treeT['StartOffset'] = group['TreeHalosOffset'][()]
            # treeT['TreeID'] = group['ForestID'][()]

            # treeT['Header'] = {key:int(val) for key, val in tr['Header'].attrs.items()}

    return treeT

def read_tree(fname, Nfiles, sl = None, newFields = []):
    """
    Read a consistent tree hdf5 tree. If sl is not Note, it is intepreted as a slice of the
    file on the disk.

    """
    dset_names = {'TreeMainProgenitor':'FirstProgenitor',
                  'TreeDescendant':'Descendant',
                  'TreeFirstProgenitor':'FirstProgenitor',
                  'SnapNum':'Snap_idx',
                  'SubhaloMass':'Mvir',
                  'SubhaloNr':'id',
                  'Group_M_Crit200':'M200c',
                  'a':'scale',
                  # 'TreeID',
                  # 'SubhaloMassType',
                  'Px':'x',
                  'Py':'y',
                  'Pz':'z',
                  'TreeNextProgenitor':'NextProgenitor',
                  'TreeFirstHaloInFOFgroup':'FirstHaloInFOFgroup'}

    # if type(newFields) is not list:
        # raise ValueError('newFields needs to be a list')

    # dset_names.extend(['Subhalo'+key for key in newFields])


    tree = {}

    if sl is None:
        mysl = np.s_[()]
    else:
        mysl = sl

    if Nfiles > 1 or force_multiple:
        raise NotImplementedError

        # allocate the arrays
        with h5py.File(fname.with_suffix('.0.hdf5'), 'r') as tr:
            NhaloTot = tr['Header'].attrs['Nhalos_Total']

            for dset in dset_names:
                ndims = tr['/TreeHalos/'+dset].ndim

                if ndims == 1:
                    tree[dset] = np.zeros(NhaloTot,
                                   dtype=tr['/TreeHalos/'+dset].dtype)
                elif ndims == 2:
                    tree[dset] = np.zeros((NhaloTot,
                                    tr['/TreeHalos/'+dset].shape[1]),
                                    dtype=tr['/TreeHalos/'+dset].dtype)
                elif ndims == 3:
                    tree[dset] = np.zeros((NhaloTot,
                                    tr['/TreeHalos/'+dset].shape[1],
                                    tr['/TreeHalos/'+dset].shape[2]),
                                    dtype=tr['/TreeHalos/'+dset].dtype)

        # read the arrays and put them in the dictionary
        off = 0
        for i in range(Nfiles):
            with h5py.File(fname.with_suffix(f'.{i}.hdf5'), 'r') as tr:
                size = tr['Header'].attrs['Nhalos_ThisFile']

                for key, dset in tree.items():
                    ndims = dset.ndim

                    if ndims == 1:
                        tr['/TreeHalos/'+key].read_direct(
                                dset, dest_sel = np.s_[off:off+size])
                    elif ndims == 2:
                        tr['/TreeHalos/'+key].read_direct(
                                dset, dest_sel = np.s_[off:off+size, :])

                off+= size
    else:

        with h5py.File(fname, 'r') as tr:

            for dset, dset_hdf5 in dset_names.items():
                ndims = tr['/Forests/'+dset_hdf5].ndim

                if ndims == 1:
                    tree[dset] = tr['/Forests/'+dset_hdf5][mysl]
                elif ndims == 2:
                    if sl is None:
                        tree[dset] = tr['/Forests/'+dset_hdf5][mysl]
                    else:
                        tree[dset] = tr['/Forests/'+dset_hdf5][mysl,:]

    return tree

def slice_tree(treeold, sl):

    tree = {}

    for key, el in treeold.items():
        if el.ndim == 1:
            tree[key] = el[sl]
        elif el.ndim == 2:
            tree[key] = el[sl, :]
        else:
            raise NotImplementedError

    return tree



def read_units(filename):
    """
    Read gadget4 units.

    """

    units = {}

    with h5py.File(filename, 'r') as h5file:
        units['HubbleParam'] = h5file['simulation_params'].attrs['hubble']

        units['unitMass'] = 2.e33
        units['SOLAR_MASS'] = units['unitMass']

    return units

def get_main_prog(snapNr, SubhaloNr, targetSnapNr, tree, verb=0, get_full=True,
        include_starting_point = True, raiseError = True, newFields=[]):

    if snapNr < targetSnapNr:
        raise ValueError("snapNr must be >= targetSnapNr")

    # a large positive integer number, to avoid an infinite loop below
    Nlarge = 100_000

    # using int() ensures we have an arrya of only one element
    ind = int(np.flatnonzero(
            (tree['SnapNum'] == snapNr) & (tree['SubhaloNr'] == SubhaloNr)))

    if get_full:
        outSubhaloNr = []
        outSnapNr = []
        outMass = []
        outMassDM = []
        outMassStars = []
        outMassGas = []
        outGrN = []
        outM200c = []
        outCentral = []
        out = {key:[] for key in newFields}

        if include_starting_point:
            outSubhaloNr.append(SubhaloNr)
            outSnapNr.append(snapNr)
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

    # special case
    if snapNr == targetSnapNr:
        tmp = {
                'SubhaloNr' : np.concatenate([outSubhaloNr]),
                'SnapNr' : np.concatenate([outSnapNr]),
                'Mass' : np.concatenate([outMass]),
                'GrN' : np.concatenate([outGrN]),
                'M200c' : np.concatenate([outM200c]),
                'MassDM' : np.concatenate([outMassDM]),
                'MassStars' : np.concatenate([outMassStars]),
                'MassGas' : np.concatenate([outMassGas]),
                'central' : np.concatenate([outCentral])
                }
        for key in out:
            tmp[key] = np.array(out[key])

        return tmp

    break_invalid = False

    for j in range(Nlarge):

        ind2 = tree['TreeMainProgenitor'][ind]

        # invalid index
        if ind2 < 0:
            if tree['TreeFirstProgenitor'][ind] >=0:
                print('WARNING: Main progenitor not'
                      ' found, but first progenitor is present')
            break_invalid = True
            if raiseError:
                raise CTNoProgenitorError
            break

        if targetSnapNr == tree['SnapNum'][ind2]:
            ind = ind2
            break

        ind = ind2

        if get_full:
            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])


    if get_full:

        if not break_invalid:
            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

        tmp = {
                'SubhaloNr' : np.concatenate([outSubhaloNr]),
                'SnapNr'    : np.concatenate([outSnapNr]),
                'GrN'       : np.concatenate([outGrN]),
                'M200c'     : np.concatenate([outM200c]),
                'Mass'      : np.concatenate([outMass]),
                'MassGas'   : np.concatenate([outMassGas]),
                'MassDM'    : np.concatenate([outMassDM]),
                'MassStars' : np.concatenate([outMassStars]),
                'central'   : np.concatenate([outCentral])
                }
        for key in out:
            tmp[key] = np.array(out[key])
        return tmp
    else:
        tmp={
                'SubhaloNr' : tree['SubhaloNr'][ind],
                'SnapNr'    : tree['SnapNum'][ind],
                'GrN'       : tree['GroupNr'][ind],
                'M200c'     : tree['Group_M_Crit200'][ind],
                'Mass'      : tree['SubhaloMass'][ind],
                'MassGas'   : tree['SubhaloMassType'][ind,0],
                'MassDM'    : tree['SubhaloMassType'][ind,1],
                'MassStars' : tree['SubhaloMassType'][ind,4],
                'central'   : True if tree['TreeFirstHaloInFOFgroup'][ind] == ind else False
                }
        for key in newFields:
            tmp[key] = tree['Subhalo'+key][ind]
        return tmp



def get_main_prog_full(finalSnapN, SubhaloNr, targetSnapNr, path, allMainP =
        True, include_starting_point = True, verb = 0, raiseError = True):
    """
    Read the gadget merger tree for a galaxy and return its main progenitor
    a certain snapshot.

    NOTE: this is "expensive" since we read the merger tree at every call, use
    get_main_prog for better performance. This is a convenient wrapper when you
    have a handful of galaxies and a small run.

    finalSnapN: snapshot number of the subhalo you want to know the
                progenitors of

    SubhaloNr: subhalo position in the subhalo table

    targetSnapNr: snapshot number of the progenitor

    path: path of the tree files, can be anything that PurePath can convert

    allMainP: return all the main progenitors up to targetSnapNr

    include_starting_point: include also the final galaxy
                            if allMainP is True (ignored if
                            allMainP is False)

    """

    mypath = PurePath(path)
    TreeID = read_treelink(mypath, finalSnapN)

    tt = read_treeTable(mypath/'trees.hdf5', 1)

    # find which tree we need to load
    mask = tt['TreeID'] == TreeID[SubhaloNr]
    assert np.count_nonzero(mask) == 1

    lenght = tt['Length'][mask][0]
    off = tt['StartOffset'][mask][0]
    sl = slice(off, off+lenght)

    tree = read_tree(mypath/'trees.hdf5', 1, sl=sl)
    units = read_units(mypath/'trees.hdf5')

    # SubhaloNr_main_prog, snapN, Mass = get_main_prog(finalSnapN,
    mainProg = get_main_prog(finalSnapN,
            SubhaloNr, targetSnapNr, tree, get_full = allMainP,
            include_starting_point = include_starting_point,
            verb = verb, raiseError = raiseError)

    # convert in solar masses
    mainProg['Mass']      *=units['unitMass']/units['SOLAR_MASS']/units['HubbleParam']
    mainProg['MassGas']   *=units['unitMass']/units['SOLAR_MASS']/units['HubbleParam']
    mainProg['MassDM']    *=units['unitMass']/units['SOLAR_MASS']/units['HubbleParam']
    mainProg['MassStars'] *=units['unitMass']/units['SOLAR_MASS']/units['HubbleParam']
    mainProg['M200c']     *=units['unitMass']/units['SOLAR_MASS']/units['HubbleParam']

    return mainProg

class forestCT:
    def __init__(self, fname, newFields=['SFR', 'Pos'], readEntireForest =
                 True):
        """
        Read the gadget merger tree for a galaxy and return its main progenitor
        a certain snapshot.

        This is expensive in memory since we read ALL the merger tree
        in the initialization. This is a convenient wrapper when you have a
        lot of galaxies and complicated selections (not only one tree).

        """
        self._formatFields = [ 'M200c', 'Mass', 'MassGas', 'MassDM',
                              'MassStars', 'SFR', 'z']
        self.readEntireForest = readEntireForest

        self.treeFilename = PurePath(fname)
        self.mypath = self.treeFilename.parent
        self.Nfiles = 1
        # self.treeFilename, self.Nfiles= self._getFilename(self.mypath)
        self.newFields = newFields

        self.units = read_units(self.treeFilename)

        self.tt = read_treeTable(self.treeFilename, self.Nfiles)

        if self.readEntireForest:
            # read the full merger tree and store it
            self.tree = read_tree(self.treeFilename, self.Nfiles,
                                  newFields = self.newFields)

            self._convert_units(self.tree, self.units)

            S = np.unique(self.tree['SnapNum'])
            self.Nsnaps = S.size
            np.testing.assert_equal(S, np.arange(S.size, dtype=S.dtype))
            del S
        #else:
        #    with h5py.File(self.treeFilename, 'r') as ff:
        #        S = np.unique(ff['/Forests/Snap_idx'][()])


    @staticmethod
    def _convert_units(tree, units):

        dset_names = ['SubhaloMass', 'Group_M_Crit200', 'SubhaloMassType']

        # convert in solar masses
        for dset in dset_names:
            if dset in tree:
                tree[dset] *= (units['unitMass']
                                     /units['SOLAR_MASS']
                                     /units['HubbleParam'])


    def _convertToTable(self, dictIn, logMass=True):

        tmp = Table(dictIn, copy=False)
        if logMass:
            tmp['M200c']     = np.log10(tmp['M200c'])
            tmp['Mass']      = np.log10(tmp['Mass'])
            # tmp['MassGas']   = np.log10(tmp['MassGas'])
            # tmp['MassDM']    = np.log10(tmp['MassDM'])
            # tmp['MassStars'] = np.log10(tmp['MassStars'])
            tmp['M200c'].unit     = u.dex(u.Msun)
            tmp['Mass'].unit      = u.dex(u.Msun)
            # tmp['MassGas'].unit   = u.dex(u.Msun)
            # tmp['MassDM'].unit    = u.dex(u.Msun)
            # tmp['MassStars'].unit = u.dex(u.Msun)
        else:
            tmp['M200c'].unit = u.Msun
            tmp['Mass'].unit = u.Msun
            # tmp['MassGas'].unit = u.Msun
            # tmp['MassDM'].unit = u.Msun
            # tmp['MassStars'].unit = u.Msun


        if 'SFR' in tmp.colnames:
            tmp['SFR'].unit = u.Msun/u.yr
        if 'Pos' in list(tmp.colnames):
            del tmp['Pos']
            tmp['Coordx'] = dictIn['Pos'][:,0]
            tmp['Coordy'] = dictIn['Pos'][:,1]
            tmp['Coordz'] = dictIn['Pos'][:,2]

            tmp['Coordx'].unit = u.Mpc
            tmp['Coordy'].unit = u.Mpc
            tmp['Coordz'].unit = u.Mpc

        for key in self._formatFields:
            if key in tmp.colnames:
                tmp[key].format = '.2f'

        return tmp

    def _convertToTableFoF(self, dictIn, logMass=True):

        tmp = Table(dictIn, copy=False)
        if logMass:
            tmp['M200c']     = np.log10(tmp['M200c'])
            tmp['M200c'].unit     = u.dex(u.Msun)
        else:
            tmp['M200c'].unit = u.Msun

        for key in self._formatFields:
            if key in tmp.colnames:
                tmp[key].format = '.2f'

        return tmp

    @staticmethod
    def _getFilename(mypath):
        """
        Returns the right filename of the tree and whether it's serial or
        parallel

        """

        nameParallel = mypath /'treedata'/ f'trees.0.hdf5'
        nameSerial = mypath / f'trees.hdf5'

        if Path(nameParallel).exists():

            with h5py.File(nameParallel, 'r') as tree:
                Nfiles = tree['Header'].attrs['NumFiles']
            return nameParallel.with_suffix(''), Nfiles

        elif Path(nameSerial).exists():

            # with h5py.File(nameSerial, 'r') as tree:
                # Nfiles = tree['Header'].attrs['NumFiles']
            Nfiles = 1
            assert Nfiles == 1
            return nameSerial, Nfiles

        else:
            raise FileNotFoundError('I cannot infer the right filename'
                    f' from {mypath}, I tried {nameParallel} and'
                    f' {nameSerial}')

    def subHistory(self, indForest, targetSnapNr=0, t=False, raiseError
                   = False):
        """
        Return the history of a subhalo from a certain snapshot,
        following its main branch.



        snapN: snapshot number of the subhalo you want to know the history

        SubhaloNr: subhalo position in the subhalo table

        targetSnapNr [optional, def=0] : max snapshot number back in time

        """

        # a large positive integer number, to avoid an infinite loop below
        Nlarge = 10_000

        if indForest == 0:
            indtree = 0
        else:
            indtree=int(np.searchsorted(self.tt['StartOffset'],
                                        indForest, side='right'))-1

        # cut the tree
        sl = slice(self.tt['StartOffset'][indtree],
                   self.tt['Length'][indtree]+
                   self.tt['StartOffset'][indtree])

        if self.readEntireForest:
            tree = slice_tree(self.tree, sl)
        else:
            tree = read_tree(self.treeFilename, self.Nfiles,
                             newFields = self.newFields,
                             sl = sl)

            self._convert_units(tree, self.units)


        ind = indForest - self.tt['StartOffset'][indtree]

        outSubhaloNr = []
        outSnapNr = []
        outMass = []
        outM200c = []
        outCentral = []
        outz = []

        outSubhaloNr.append(tree['SubhaloNr'][ind])
        outSnapNr.append(tree['SnapNum'][ind])
        outMass.append(tree['SubhaloMass'][ind])
        outM200c.append(tree['Group_M_Crit200'][ind])
        outz.append(1/tree['a'][ind]-1)
        if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
            outCentral.append(True)
        else:
            outCentral.append(False)


        break_invalid = False

        for j in range(Nlarge):

            ind2 = tree['TreeMainProgenitor'][ind]

            # invalid index
            if ind2 < 0:
                if tree['TreeFirstProgenitor'][ind] >=0:
                    print('WARNING: Main progenitor not'
                          ' found, but first progenitor is present')
                break_invalid = True
                if raiseError:
                    raise CTNoProgenitorError
                break

            if targetSnapNr == tree['SnapNum'][ind2]:
                ind = ind2
                break

            ind = ind2

            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outz.append(1/tree['a'][ind]-1)

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

        if not break_invalid:
            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outz.append(1/tree['a'][ind]-1)

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

        tmp = {
                'SubhaloNr' : np.concatenate([outSubhaloNr]),
                'SnapNr'    : np.concatenate([outSnapNr]),
                'M200c'     : np.concatenate([outM200c]),
                'Mass'      : np.concatenate([outMass]),
                'central'   : np.concatenate([outCentral]),
                'z'         : np.concatenate([outz])
                }

        if t:
            return self._convertToTable(tmp)
        else:
            return tmp

    def subMain(self, snapN, SubhaloNr, targetSnapNr):
        """
        Return the main branch progenitor of a subhalo from a certain snapshot.


        snapN: snapshot number of the subhalo you want to know the progenitor

        SubhaloNr: subhalo position in the subhalo table

        targetSnapNr : snapshot number back in time

        """
        raise NotImplementedError

        # this is just a wrapper for _get_main_prog_full
        return self._get_main_prog_full(snapN, SubhaloNr, targetSnapNr,
                                       allMainP = False,
                                       include_starting_point = False,
                                       verb = 0, raiseError = True)



    def subHasMainDesc(self, snapNr, SubhaloNr, targetSnapNr = None):
        """
        Return if a subhalo at snapshot snapNr survives as a main descendant
        to snapshot targetSnapNr (default the final snapshot in the tree).

        """

        if targetSnapNr is None:
            targetSnapNr = self.tree['SnapNum'].max()

        # walk the tree down to a certain snapshot number
        try:
            desc = self.subDesc(snapNr, SubhaloNr,
                                targetSnapNr = targetSnapNr)
        except NoDescendantError:
            return False

        # make sure you get there
        assert desc['SnapNr'][-1] == targetSnapNr

        # walk back on the *main* progenitor branch
        hist = self._get_main_prog_full(desc['SnapNr'][-1],
                                       desc['SubhaloNr'][-1], snapNr,
                                       allMainP = False,
                                       verb = 0, raiseError = True)

        # if the main progenitor is the original subhalo, we have a subhalo
        # that survives down to targetSnapNr
        if hist['SnapNr'] == snapNr and hist['SubhaloNr'] == SubhaloNr:
            return True
        else:
            return False


    def subHasDesc(self, snapNr, SubhaloNr, targetSnapNr = None):
        """
        Return if a subhalo at snapshot snapNr survives as a descendant
        to snapshot targetSnapNr (default the final snapshot in the tree).
        Useful to spot subhaloes that are not tracked down to targetSnapNr.

        """
        SubhaloNr = np.atleast_1d(SubhaloNr)
        SubhaloNr = np.array(SubhaloNr)

        if targetSnapNr is None:
            targetSnapNr = self.tree['SnapNum'].max()

        if targetSnapNr>self.Nsnaps-1:
            raise DescendantSnapNrError(f'targetSnapNr must be <= of {self.Nsnaps-1}')


        # find which tree we need to load, assume they are
        # all in the same tree
        mask = self.tt['TreeID'] == self.TreeID[snapNr][SubhaloNr[0]]
        assert np.count_nonzero(mask) == 1

        lenght = self.tt['Length'][mask][0]
        off = self.tt['StartOffset'][mask][0]
        sl = slice(off, off+lenght)

        # tree = slice_tree(self.tree, sl)

        # slice it manually, we don't need all the fields
        tree = {'SnapNum':self.tree['SnapNum'][sl],
                'SubhaloNr':self.tree['SubhaloNr'][sl],
                'TreeDescendant':self.tree['TreeDescendant'][sl]}

        # a large positive integer number, to avoid an infinite loop below
        Nlarge = 10_000

        # special case
        if snapNr == targetSnapNr:
            return True

        out = np.zeros_like(SubhaloNr, dtype=bool)

        for k, sNr in enumerate(SubhaloNr):
            # using int() ensures we have an array of only one element
            ind = int(np.flatnonzero(
                    (tree['SnapNum'] == snapNr) & (tree['SubhaloNr'] == sNr)))


            for j in range(Nlarge):

                ind2 = tree['TreeDescendant'][ind]

                # invalid index
                if ind2 < 0:
                    # it's false by default
                    # out[k] = False
                    break

                if targetSnapNr == tree['SnapNum'][ind2]:
                    out[k] = True
                    break

                ind = ind2

        # outtmp = tmpfunc(tree['TreeDescendant'], tree['SnapNum'],
                     # tree['SubhaloNr'], snapNr, SubhaloNr, targetSnapNr)
        # np.testing.assert_equal(out, outtmp)

        return out



    def subDesc(self, snapNr, SubhaloNr, targetSnapNr = None,
                raiseError = True, verb = False, t = False):
        """
        Return the future history of a subhalo from a certain snapshot.


        snapNr: snapshot number of the subhalo you want to know the history

        SubhaloNr: subhalo position in the subhalo table

        targetSnapNr [optional] : max snapshot number forward in time

        """
        if targetSnapNr is None:
            targetSnapNr = self.tree['SnapNum'].max()

        if targetSnapNr>self.Nsnaps-1:
            raise DescendantSnapNrError(f'targetSnapNr must be <= of {self.Nsnaps-1}')


        # find which tree we need to load
        mask = self.tt['TreeID'] == self.TreeID[snapNr][SubhaloNr]
        assert np.count_nonzero(mask) == 1

        lenght = self.tt['Length'][mask][0]
        off = self.tt['StartOffset'][mask][0]
        sl = slice(off, off+lenght)

        tree = slice_tree(self.tree, sl)


        # a large positive integer number, to avoid an infinite loop below
        Nlarge = 100_000

        # using int() ensures we have an array of only one element
        ind = int(np.flatnonzero(
                (tree['SnapNum'] == snapNr) & (tree['SubhaloNr'] == SubhaloNr)))

        outSubhaloNr = []
        outSnapNr = []
        outMass = []
        outMassDM = []
        outMassStars = []
        outMassGas = []
        outGrN = []
        outM200c = []
        outCentral = []
        out = {key:[] for key in self.newFields}

        outSubhaloNr.append(SubhaloNr)
        outSnapNr.append(snapNr)
        outMass.append(tree['SubhaloMass'][ind])
        outGrN.append(tree['GroupNr'][ind])
        outM200c.append(tree['Group_M_Crit200'][ind])
        outMassDM.append(tree['SubhaloMassType'][ind,1])
        outMassStars.append(tree['SubhaloMassType'][ind,4])
        outMassGas.append(tree['SubhaloMassType'][ind,0])
        if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
            outCentral.append(True)
        else:
            outCentral.append(False)

        for key in out:
            out[key].append(tree['Subhalo'+key][ind])

        # special case
        if snapNr == targetSnapNr:
            tmp = {
                    'SubhaloNr' : np.concatenate([outSubhaloNr]),
                    'SnapNr' : np.concatenate([outSnapNr]),
                    'Mass' : np.concatenate([outMass]),
                    'GrN' : np.concatenate([outGrN]),
                    'M200c' : np.concatenate([outM200c]),
                    'MassDM' : np.concatenate([outMassDM]),
                    'MassStars' : np.concatenate([outMassStars]),
                    'MassGas' : np.concatenate([outMassGas]),
                    'central' : np.concatenate([outCentral])
                    }
            for key in out:
                tmp[key] = np.array(out[key])

            return tmp

        break_invalid = False

        for j in range(Nlarge):

            ind2 = tree['TreeDescendant'][ind]

            # invalid index
            if ind2 < 0:
                break_invalid = True
                if raiseError:
                    raise NoDescendantError
                break

            if targetSnapNr == tree['SnapNum'][ind2]:
                ind = ind2
                break

            ind = ind2

            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

        if not break_invalid:
            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])

            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

        desc = {
                'SubhaloNr' : np.concatenate([outSubhaloNr]),
                'SnapNr'    : np.concatenate([outSnapNr]),
                'GrN'       : np.concatenate([outGrN]),
                'M200c'     : np.concatenate([outM200c]),
                'Mass'      : np.concatenate([outMass]),
                'MassGas'   : np.concatenate([outMassGas]),
                'MassDM'    : np.concatenate([outMassDM]),
                'MassStars' : np.concatenate([outMassStars]),
                'central'   : np.concatenate([outCentral])
                }

        for key in out:
            desc[key] = np.array(out[key])


        desc['z'] = self.tt['z'][desc['SnapNr']]

        if t:
            return self._convertToTable(desc)
        else:
            return desc

    def subLastMainDesc(self, snapNr, SubhaloNr):
        """
        Return the last main descendant of a subhalo from a certain snapshot.


        snapNr: snapshot number of the subhalo you want to know the history

        SubhaloNr: subhalo position in the subhalo table

        """
        targetSnapNr = self.tree['SnapNum'].max()

        if targetSnapNr>self.Nsnaps-1:
            raise DescendantSnapNrError(f'targetSnapNr must be <= of {self.Nsnaps-1}')


        # find which tree we need to load
        mask = self.tt['TreeID'] == self.TreeID[snapNr][SubhaloNr]
        assert np.count_nonzero(mask) == 1

        lenght = self.tt['Length'][mask][0]
        off = self.tt['StartOffset'][mask][0]
        sl = slice(off, off+lenght)

        tree = slice_tree(self.tree, sl)


        # a large positive integer number, to avoid an infinite loop below
        Nlarge = 100_000

        # using int() ensures we have an array of only one element
        ind = int(np.flatnonzero(
                (tree['SnapNum'] == snapNr) & (tree['SubhaloNr'] == SubhaloNr)))

        outSubhaloNr = []
        outSnapNr = []
        outMass = []
        outMassDM = []
        outMassStars = []
        outMassGas = []
        outGrN = []
        outM200c = []
        outCentral = []
        out = {key:[] for key in self.newFields}

        # special case
        if snapNr == targetSnapNr:
            outSubhaloNr.append(SubhaloNr)
            outSnapNr.append(snapNr)
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

            tmp = {
                    'SubhaloNr' : np.concatenate([outSubhaloNr]),
                    'SnapNr' : np.concatenate([outSnapNr]),
                    'Mass' : np.concatenate([outMass]),
                    'GrN' : np.concatenate([outGrN]),
                    'M200c' : np.concatenate([outM200c]),
                    'MassDM' : np.concatenate([outMassDM]),
                    'MassStars' : np.concatenate([outMassStars]),
                    'MassGas' : np.concatenate([outMassGas]),
                    'central' : np.concatenate([outCentral])
                    }

            for key in out:
                tmp[key] = np.array(out[key])

            return tmp

        for j in range(Nlarge):

            ind2 = tree['TreeDescendant'][ind]

            # 3 possible outcomes:

            # 1) invalid index, the last one is the main
            if ind2 < 0:
                break

            # 2) reached the end
            # 3) got a valid descendant
            # the last two need to be checked

            # walk back on the *main* progenitor branch
            indMainP = tree['TreeMainProgenitor'][ind2]

            if indMainP != ind:
                break

            # the main prog of the descendant is the original subhalo,
            # keep going
            ind = ind2

            # but stop if the descendant is valid AND at the last snapshot
            if targetSnapNr == tree['SnapNum'][ind2]:
                break


        outSubhaloNr.append(tree['SubhaloNr'][ind])
        outSnapNr.append(tree['SnapNum'][ind])
        outMass.append(tree['SubhaloMass'][ind])
        outGrN.append(tree['GroupNr'][ind])
        outM200c.append(tree['Group_M_Crit200'][ind])
        outMassGas.append(tree['SubhaloMassType'][ind,0])
        outMassDM.append(tree['SubhaloMassType'][ind,1])
        outMassStars.append(tree['SubhaloMassType'][ind,4])

        if tree['TreeFirstHaloInFOFgroup'][ind] == ind:
            outCentral.append(True)
        else:
            outCentral.append(False)

        for key in out:
            out[key].append(tree['Subhalo'+key][ind])

        desc = {
                'SubhaloNr' : np.concatenate([outSubhaloNr]),
                'SnapNr'    : np.concatenate([outSnapNr]),
                'GrN'       : np.concatenate([outGrN]),
                'M200c'     : np.concatenate([outM200c]),
                'Mass'      : np.concatenate([outMass]),
                'MassGas'   : np.concatenate([outMassGas]),
                'MassDM'    : np.concatenate([outMassDM]),
                'MassStars' : np.concatenate([outMassStars]),
                'central'   : np.concatenate([outCentral])
                }

        for key in out:
            desc[key] = np.array(out[key])

        desc['z'] = self.tt['z'][desc['SnapNr']]

        return desc

    def subImmediateProgs(self, SnapN, SubhaloNr, t=False):
        """
        Return all the immediate progenitors of a subhalo.

        SnapN: snapshot number of the subhalo you want to know the
                    progenitors of

        SubhaloNr: subhalo position in the subhalo table

        """

        if SnapN>self.Nsnaps-1:
            raise ValueError(f'SnapN must be < of {self.Nsnaps-1}')

        # find which tree we need to load
        mask = self.tt['TreeID'] == self.TreeID[SnapN][SubhaloNr]
        assert np.count_nonzero(mask) == 1

        lenght = self.tt['Length'][mask][0]
        off = self.tt['StartOffset'][mask][0]
        sl = slice(off, off+lenght)

        tree = slice_tree(self.tree, sl)

        # identify the position of the subhalo in the tree
        # using int() ensures we have an array of only one element
        ind = int(np.flatnonzero(
                (tree['SnapNum'] == SnapN) &
                (tree['SubhaloNr'] == SubhaloNr)))

        progs_list = forest._get_progs_worker(ind, tree)

        # concatenate all the lists
        progs = {key: np.concatenate([el]) for key, el in progs_list.items()}

        progs['z'] = self.tt['z'][progs['SnapNr']]

        if t:
            return self._convertToTable(progs)
        else:
            return progs

    def subAllProgs(self, indForest, t=False, onlyIndTree=False):
        """
        Return all the progenitors of a subhalo up to a certain z.
        Note that due to the way Gadget 4 builds merger tree, this is likely
        to be a subsample of the tree (subhaloes within the same FoF are
        grouped together in the tree by Gadget 4).

        SnapN: snapshot number of the subhalo you want to know the
                    progenitors of

        SubhaloNr: subhalo position in the subhalo table

        """


        if indForest == 0:
            indtree = 0
        else:
            indtree=int(np.searchsorted(self.tt['StartOffset'],
                                        indForest, side='right'))-1

        # cut the tree
        sl = slice(self.tt['StartOffset'][indtree],
                   self.tt['Length'][indtree]+
                   self.tt['StartOffset'][indtree])
        if self.readEntireForest:
            tree = slice_tree(self.tree, sl)
        else:
            tree = read_tree(self.treeFilename, self.Nfiles,
                             newFields = self.newFields,
                             sl = sl)

            self._convert_units(tree, self.units)

        # identify the position of the subhalo in the tree
        # using int() ensures we have an array of only one element
        ind = indForest - self.tt['StartOffset'][indtree]

        # walk along the tree with a recursive function and save only the
        # indexes of all the progenitors
        # NOTE: as long as the entire tree fits in memory, this is fast since I
        # don't build lists and concatenate them except for the index one
        indTree = np.array(forestCT._get_progs_workerRec(ind, tree))
        assert all(indTree >= 0)

        if onlyIndTree:
            return indTree

        # the first subhalo in a FoF has TreeFirstHaloInFOFgroup pointing to
        # itself
        central = np.zeros_like(indTree, dtype=bool)
        central[np.equal(tree['TreeFirstHaloInFOFgroup'][indTree], indTree) ] = True

        progs = {
                'SubhaloNr' : tree['SubhaloNr'][indTree],
                'SnapNr'    : tree['SnapNum'][indTree],
                # 'GrN'       : tree['GroupNr'][indTree],
                'M200c'     : tree['Group_M_Crit200'][indTree],
                'Mass'      : tree['SubhaloMass'][indTree],
                # 'MassGas'   : tree['SubhaloMassType'][indTree,0],
                # 'MassDM'    : tree['SubhaloMassType'][indTree,1],
                # 'MassStars' : tree['SubhaloMassType'][indTree,4],
                'z'           : 1/tree['a'][indTree]-1,
                'TreeInd'   : indTree,
                'central'   : central
                }

        # progs['z'] = self.tt['z'][progs['SnapNr']]

        if t:
            return self._convertToTable(progs)
        else:
            return progs

    @staticmethod
    def _get_progs_workerRec(ind_in, tree):

        # ----------
        # get the first progenitor
        ind = tree['TreeFirstProgenitor'][ind_in]

        # invalid index
        if ind < 0:
            raise NoFirstProgenitorError

        Nlarge = 100_000

        outTreeInd = [ind]

        # this loop is not over the snapshots but over the progs
        for j in range(Nlarge):

            ind = tree['TreeNextProgenitor'][ind]

            # invalid index, we have reached the end of progenitors
            if ind < 0:
                break

            outTreeInd.append(ind)

        if j >= Nlarge-1:
            raise OutOfRangeLoopError

        # ----------

        # do all over again with the results I have obtained
        tmp = []
        for i in outTreeInd:
            try:
                ind2 = forestCT._get_progs_workerRec(i, tree)
                tmp.extend(ind2)
                del ind2
            except NoFirstProgenitorError:
                pass

        outTreeInd.extend(tmp)
        return outTreeInd


    @staticmethod
    def _get_progs_worker(ind, tree, newFields=[]):
        """
        Get all the immediate progenitors of a subhalo.

        """

        # get the first progenitor
        indFirstP = tree['TreeFirstProgenitor'][ind]

        # invalid index
        if indFirstP < 0:
            raise NoFirstProgenitorError

        outSubhaloNr = [tree['SubhaloNr'][indFirstP]]
        outSnapNr = [tree['SnapNum'][indFirstP]]
        outMass = [tree['SubhaloMass'][indFirstP]]
        outGrN = [tree['GroupNr'][indFirstP]]
        outM200c = [tree['Group_M_Crit200'][indFirstP]]
        outMassGas = [tree['SubhaloMassType'][indFirstP,0]]
        outMassDM = [tree['SubhaloMassType'][indFirstP,1]]
        outMassStars = [tree['SubhaloMassType'][indFirstP,4]]
        outTreeInd = [indFirstP]
        if tree['TreeFirstProgenitor'][ind] == ind:
            outCentral = [True]
        else:
            outCentral = [False]

        out = {key:[tree['Subhalo'+key][indFirstP]] for key in newFields}

        # a large positive integer number,
        # to avoid an infinite loop below
        Nlarge = 100_000

        ind = indFirstP

        # this loop is not over the snapshots but over the progs
        for j in range(Nlarge):

            ind = tree['TreeNextProgenitor'][ind]

            # invalid index, we have reached the end of progenitors
            if ind < 0:
                break


            outSubhaloNr.append(tree['SubhaloNr'][ind])
            outSnapNr.append(tree['SnapNum'][ind])
            outMass.append(tree['SubhaloMass'][ind])
            outGrN.append(tree['GroupNr'][ind])
            outM200c.append(tree['Group_M_Crit200'][ind])
            outMassGas.append(tree['SubhaloMassType'][ind,0])
            outMassDM.append(tree['SubhaloMassType'][ind,1])
            outMassStars.append(tree['SubhaloMassType'][ind,4])
            outTreeInd.append(ind)
            if tree['TreeFirstProgenitor'][ind] == ind:
                outCentral.append(True)
            else:
                outCentral.append(False)

            for key in out:
                out[key].append(tree['Subhalo'+key][ind])

        if j >= Nlarge-1:
            raise OutOfRangeLoopError

        tmp = {
                'SubhaloNr' : outSubhaloNr,
                'SnapNr'    : outSnapNr,
                'GrN'       : outGrN,
                'M200c'     : outM200c,
                'Mass'      : outMass,
                'MassGas'   : outMassGas,
                'MassDM'    : outMassDM,
                'MassStars' : outMassStars,
                'TreeInd'   : outTreeInd,
                'central'   : outCentral
                }
        for key in out:
            tmp[key] = out[key]

        return tmp
