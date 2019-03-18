"""
protein_features.py
This is a tool to featurize kinase conformational changes through the entire Kinome.

"""

# Setup general logging (guarantee output/error message in case of interruption)
# TODO: Can we log to the terminal instead?
import logging
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)


def key_klifs_residues(numbering):
    """
    Retrieve a list of PDB residue indices relevant to key kinase conformations mapped via KLIFS.

    Define indices of the residues relevant to a list of 12 collective variables relevant to
    kinase conformational changes. These variables include: angle between aC and aE helices,
    the key K-E salt bridge, DFG-Phe conformation (two distances), X-DFG-Phi, X-DFG-Psi,
    DFG-Asp-Phi, DFG-Asp-Psi, DFG-Phe-Phi, DFG-Phe-Psi, DFG-Phe-Chi1, and the FRET L-S distance.
    All features are under the current numbering of the structure provided.

    Parameters
    ----------
    numbering : list of int
        numbering[klifs_index] is the residue number for the given PDB file corresponding to KLIFS residue index 'klifs_index'

    Returns
    -------
    key_res : list of int
        Key residue indices

    """

    key_res = []
    # angle between aC and aE helices
    key_res.append(numbering[20])  # residue 21 (res1 in aC)
    key_res.append(numbering[28])  # res29 (res2 in aC)
    key_res.append(numbering[60])  # res61 (res1 in aE)
    key_res.append(numbering[62])  # res63 (res2 in aE)

    # key salt bridge
    key_res.append(numbering[16])  # res17 (K in beta3)
    key_res.append(numbering[23])  # res24 (E in aC)

    # DFG conformation and Phe conformation
    key_res.append(numbering[27])  # res28 (ExxxX)
    key_res.append(numbering[81])  # res82 (DFG-Phe)

    # X-DFG Phi/Psi
    key_res.append(numbering[79])  # res80 (X-DFG)

    # DFG-Asp Phi/Psi
    key_res.append(numbering[80])  # res81 (DFG-Asp)

    # FRET distance
    # not in the list of 85 (equivalent to Aura"S284"), use the 100% conserved beta III K as a reference
    key_res.append(numbering[16] + 120)

    # not in the list of 85 (equivalent to Aura"L225"), use the 100% conserved beta III K as a reference
    key_res.append(numbering[16] + 61)

    return key_res

def compute_simple_protein_features(pdbid, chainid, coordfile, numbering):
    """
    This function takes the PDB code, chain id and certain coordinates of a kinase from
    a command line and returns its structural features.

    Parameters
    ----------
    pdbid : str
        The PDB code of the inquiry kinase.
    chainid : str
        The chain index of the inquiry kinase.
    coordfile : str
        Specifies the file constaining the kinase coordinates (either a pdb file or a trajectory, i.e. trj, dcd, h5)
    numbering : list of int
        The residue indices of the 85 pocket residues specific to the structure.

    Returns
    -------
    dihedrals: list of floats
        A list (one frame) or lists (multiple frames) of dihedrals relevant to kinase conformation.
    distances: list of floats
        A list (one frame) or lists (multiple frames) of intramolecular distances relevant to kinase conformation.

    .. todo ::

       Instead of featurizing on dihedrals (which are discontinuous), it's often better to use sin() and cos()
       of the dihedrals or some other non-discontinuous representation.

    .. todo :: Instead of a PDB file or a trj/dcd/h5, accept an MDTraj.Trajectory---this will be much more flexible.

    .. todo :: Use kwargs with sensible defaults instead of relying only on positional arguments.


    """
    import tempfile
    import os
    import mdtraj as md
    import numpy as np

    pdb_file = None

    # A safer way to download files as wget may not exist on systems such MacOS
    # TODO: Since we retrieve the PDB file in multiple pieces of code, let's refactor this into one utility function
    # to avoid code duplication.
    import urllib

    # get toppology info either from fixed pdb or original pdb file (based on input) 
    # if analyzing a trajectory 
    if coordfile == 'dcd':
        traj = md.load(str(pdbid) + '.dcd',top = str(pdbid) + '_fixed_solvated.pdb')
        topology = md.load(str(pdbid)+'_fixed.pdb').topology

    with urllib.request.urlopen('http://www.pdb.org/pdb/files/{}.pdb'.format(pdbid)) as response:
        pdb_file = response.read()

    with tempfile.TemporaryDirectory() as pdb_directory:
        pdb = os.path.join(pdb_directory,'{}.pdb'.format(pdbid))
        with open(pdb, 'w') as file:
            file.write(pdb_file.decode())
            # load traj before the temp pdb file was removed
            if coordfile == 'pdb':
                print("loading top from pdb")
                traj = md.load(pdb)
                topology = md.load(pdb).topology
    table, bonds = topology.to_dataframe()
    atoms = table.values
    # translate a letter chain id into a number index (A->0, B->1 etc)
    # TODO: This may not be robust, since chains aren't always in sequence from A to Z
    chain_index = ord(str(chainid).lower()) - 97

    # get the array of atom indices for the calculation of:
    #       * eight dihedrals (a 8*4 array where each row contains indices of the four atoms for each dihedral)
    #       * five ditances (a 5*2 array where each row contains indices of the two atoms for each dihedral)
    dih = np.zeros(shape=(8, 4), dtype=int, order='C')
    dis = np.zeros(shape=(5, 2), dtype=int, order='C')

    # name list of the dihedrals and distances
    dih_names = [
        'aC_rot', 'xDFG_phi', 'xDFG_psi', 'dFG_phi', 'dFG_psi', 'DfG_phi',
        'DfG_psi', 'DfG_chi'
    ]
    dis_names = ['K_E1', 'K_E2', 'DFG_conf1', 'DFG_conf2', 'fret']

    # parse the topology info
    '''
    The coordinates are located by row number (usually is atom index minus one, which is also why it's zero-based)
    by mdtraj but when the atom indices are not continuous there is a problem so a safer way to locate the coordinates
    is through row number (as a fake atom index) in case the atom indices are not continuous.
    '''
    count = 0  # keep track of row indices as "atom indices"
    for line in atoms:
        # for the specified chain
        if line[5] == chain_index:
            # TODO: Use MDTraj DSL instead of this cumbersome comparison scheme.
            # http://mdtraj.org/latest/atom_selection.html

            # dihedral 1: between aC and aE helices
            dih[0][0] = count if line[3] == numbering[20] and line[
                1] == 'CA' else dih[0][0]
            dih[0][1] = count if line[3] == numbering[28] and line[
                1] == 'CA' else dih[0][1]
            dih[0][2] = count if line[3] == numbering[60] and line[
                1] == 'CA' else dih[0][2]
            dih[0][3] = count if line[3] == numbering[62] and line[
                1] == 'CA' else dih[0][3]
            # dihedral 2 & 3: X-DFG Phi & Psi
            dih[1][0] = count if line[3] == numbering[78] and line[
                1] == 'C' else dih[1][0]
            dih[1][1] = count if line[3] == numbering[79] and line[
                1] == 'N' else dih[1][1]
            dih[1][2] = count if line[3] == numbering[79] and line[
                1] == 'CA' else dih[1][2]
            dih[1][3] = count if line[3] == numbering[79] and line[
                1] == 'C' else dih[1][3]
            dih[2][0] = dih[1][1]
            dih[2][1] = dih[1][2]
            dih[2][2] = dih[1][3]
            dih[2][3] = count if line[3] == numbering[80] and line[
                1] == 'N' else dih[2][3]

            # dihedral 4 & 5: DFG-Asp Phi & Psi
            dih[3][0] = dih[1][3]
            dih[3][1] = dih[2][3]
            dih[3][2] = count if line[3] == numbering[80] and line[
                1] == 'CA' else dih[3][2]
            dih[3][3] = count if line[3] == numbering[80] and line[
                1] == 'C' else dih[3][3]
            dih[4][0] = dih[3][1]
            dih[4][1] = dih[3][2]
            dih[4][2] = dih[3][3]
            dih[4][3] = count if line[3] == numbering[81] and line[
                1] == 'N' else dih[4][3]

            # dihedral 6 & 7: DFG-Phe Phi & Psi
            dih[5][0] = dih[3][3]
            dih[5][1] = dih[4][3]
            dih[5][2] = count if line[3] == numbering[81] and line[
                1] == 'CA' else dih[5][2]
            dih[5][3] = count if line[3] == numbering[81] and line[
                1] == 'C' else dih[5][3]
            dih[6][0] = dih[5][1]
            dih[6][1] = dih[5][2]
            dih[6][2] = dih[5][3]
            dih[6][3] = count if line[3] == numbering[82] and line[
                1] == 'N' else dih[6][3]

            # dihedral 8: DFG-Phe Chi
            dih[7][0] = dih[5][1]
            dih[7][1] = dih[5][2]
            dih[7][2] = count if line[3] == numbering[81] and line[
                1] == 'CB' else dih[7][2]
            dih[7][3] = count if line[3] == numbering[81] and line[
                1] == 'CG' else dih[7][3]

            # distance 1 & 2: K-E salt bridge
            dis[0][0] = count if line[3] == numbering[16] and line[
                1] == 'NZ' else dis[0][0]
            dis[0][1] = count if line[3] == numbering[23] and line[
                1] == 'OE1' else dis[0][1]
            dis[1][0] = count if line[3] == numbering[16] and line[
                1] == 'NZ' else dis[1][0]
            dis[1][1] = count if line[3] == numbering[23] and line[
                1] == 'OE2' else dis[1][1]

            # distance 3 & 4: DFG conformation-related distances
            dis[2][0] = count if line[3] == numbering[27] and line[
                1] == 'CA' else dis[2][0]
            dis[2][1] = count if line[3] == numbering[81] and line[
                1] == 'CZ' else dis[2][1]
            dis[3][0] = count if line[3] == numbering[16] and line[
                1] == 'CA' else dis[3][0]
            dis[3][1] = dis[2][1]

            # distance 5: FRET distance
            dis[4][0] = count if line[3] == int(
                numbering[80] + 10) and line[1] == 'CA' else dis[4][0]
            dis[4][1] = count if line[3] == int(
                numbering[80] - 20) and line[1] == 'CA' else dis[4][1]

        if line[5] > chain_index:
            break

        count += 1
    # check if there is any missing coordinates; if so, skip dihedral/distance calculation for those residues
    check_flag = 1
    for i in range(len(dih)):
        if 0 in dih[i]:
            dih[i] = [0,0,0,0]
            #logging.info(
            #    'The "' + str(dih_names[i]) +
            #    '" dihedral will not be computed due to missing coordinates.')
            check_flag = 0
    for i in range(len(dis)):
        if 0 in dis[i]:
            dis[i] = [0,0]
            #logging.info(
            #    'The "' + str(dis_names[i]) +
            #    '" distance will not be calculated due to missing coordinates.'
            #)
            check_flag = 0
    #if check_flag:
        #logging.info(
        #    "There is no missing coordinates.  All dihedrals and distances will be computed."
        #)
    # calculate the dihedrals and distances for the user-specifed structure (a static structure or an MD trajectory)
    dihedrals = md.compute_dihedrals(traj, dih)
    distances = md.compute_distances(traj, dis)

    # option to log the results
    '''
    logging.info("Key dihedrals relevant to kinase conformation are as follows:")
    logging.info(dih_names)
    #logging.info(dihedrals/np.pi*180) # dihedrals in degrees
    logging.info(dihedrals)  # dihedrals in radians
    logging.info("Key distances relevant to kinase conformation are as follows:")
    logging.info(dis_names)
    logging.info(distances)
    '''

    # clean up
    # TODO: This is dangerous! Instead, rely on using the "with tempfile.TemporaryDirectory" context manager idiom to create and clean up temporary directories
    #import subprocess
    #rm_file = 'rm ./' + str(pdb) + '.pdb*'
    #rm_file = 'rm ./' + str(pdb) + '.pdb*'
    #subprocess.call(rm_file, shell=True)

    del traj, dih, dis

    return dihedrals, distances
