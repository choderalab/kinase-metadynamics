"""
protein_features.py
This is a tool to featurize kinase conformational changes through the entire Kinome.

"""
import sys
import requests
import ast
import urllib.request
import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
import mdtraj as md
import subprocess


def basics(pdb, chain):
    """
    This function takes the PDB code and chain id of a kinase from a command line and returns its basic information.
    
    Parameters
    ----------
    pdb: str
        The PDB code of the inquiry kinase.
    chain: str
        The chain index of the inquiry kinase.

    Returns
    -------
    kinase_id: int
        The standard ID of a kinase enforced by the KLIFS database.
    name: str
        The standard name of the kinase used by the KLIFS database.
    pocket_seq: str
        The 85 discontinuous residues (from multisequence alignment) that define the binding pocket of a kinase.
    struct_id: int
        The ID associated with a specific chain in the pdb structure of a kinase.
    numbering: list of int
        The residue indices of the 85 pocket residues specific to the structure.
    key_res: list of int
        A list of residue indices that are relevant to the collective variables.
    
    """

    # make sure the input format is expect
    if type(pdb)!= str or type(chain) != str:
        raise ValueError("The input must be a string (PDB_code,chain_id).")

    # get information of the query kinase from the KLIFS database and gives values of kinase_id, name and pocket_seq (numbering)
    url = "http://klifs.vu-compmedchem.nl/api/structures_pdb_list?pdb-codes=" + str(
        pdb)  # form the query command
    clean = requests.get(url).text.replace('true', 'True').replace(
        'false', 'False')  # clean up the info from KLIFS
    for structure in ast.literal_eval(
            clean):  # each pdb code corresponds to multiple structures
        if structure['chain'] == str(
                chain):  # find the specific chain
            kinase_id = int(structure['kinase_ID'])
            name = str(structure['kinase'])
            pocket_seq = str(structure['pocket'])
            struct_id = int(structure['structure_ID'])

    # Get the numbering of the 85 pocket residues
    print("Getting the numbering of 85 residues...")
    cmd = "http://klifs.vu-compmedchem.nl/details.php?structure_id=" + str(
        struct_id)
    preload = urllib.request.urlopen(cmd)
    info = urllib.request.urlopen(cmd)
    for line_number, line in enumerate(info):
        line = line.decode().replace('_', '')
        if 'pocketResidues=[' in line:
            numbering = ast.literal_eval(
                (line[line.find('=') + 1:line.find(';')]))
            break
    print("Done getting residues.")
    # check if there is gaps/missing residues among the pocket residues. If so, enforce their indices as 0 and avoid using them to compute collective variables.
    for i in range(len(numbering)):
        if numbering[i] == -1:
            #print(
            #    "Warning: There is a gap/missing residue at position: " +
            #    str(i + 1) +
            #    ". Its index will be enforced as 0 and it will not be used to compute collective variables."
            #)
            numbering[i] = 0
    '''
    # define indices of the residues relevant to a list of 12 collective variables relevant to kinase conformational changes. These variables include: angle between aC and aE helices, the key K-E salt bridge, DFG-Phe conformation (two distances), X-DFG-Phi, X-DFG-Psi, DFG-Asp-Phi, DFG-Asp-Psi, DFG-Phe-Phi, DFG-Phe-Psi, DFG-Phe-Chi1, and the FRET L-S distance. All features are under the current numbering of the structure provided.
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
    key_res.append(
        numbering[16] + 120
    )  # not in the list of 85 (equivalent to Aura"S284"), use the 100% conserved beta III K as a reference

    key_res.append(
        numbering[16] + 61
    )  # not in the list of 85 (equivalent to Aura"L225"), use the 100% conserved beta III K as a reference

    
    # print out kinase information
    #print("---------------------Results----------------------")
    #print("Kinase ID: " + str(kinase_id))
    #print("Kinase name: " + str(name))
    #print("Pocket residues: " + str(pocket_seq))
    #print("Structure ID: " + str(struct_id))
    #print("Numbering of the 85 pocket residues: " + str(numbering))
    #print("Residues involved in collective variables: " + str(key_res))
    '''
    #return kinase_id, name, struct_id, pocket_seq, numbering, key_res
    return numbering

def features(pdb, chain, numbering):
    """
    This function takes the PDB code, chain id and certain coordinates of a kinase from a command line and returns its structural features.
    
    Parameters
    ----------
    pdb: str
        The PDB code of the inquiry kinase.
    chain: str
        The chain index of the inquiry kinase.
    coord: str
        Specifies the file constaining the kinase coordinates (either a pdb file or a trajectory, i.e. trj, dcd, h5)
    numbering: list of int
        The residue indices of the 85 pocket residues specific to the structure.    

    Returns
    -------
    dihedrals: list of floats
        A list (one frame) or lists (multiple frames) of dihedrals relevant to kinase conformation.
    distances: list of floats
        A list (one frame) or lists (multiple frames) of intramolecular distances relevant to kinase conformation.
    
    """

    # download the pdb structure
    cmd = 'wget -q http://www.pdb.org/pdb/files/' + str(
        pdb) + '.pdb'
    subprocess.call(cmd, shell=True)

    # get topology info from the structure
    topology = md.load(str(pdb) + '_fixed.pdb').topology
    table, bonds = topology.to_dataframe()
    atoms = table.values
    #np.set_printoptions(threshold=np.nan)
    #print(atoms)
    chain_index = ord(str(chain).lower(
    )) - 97  # translate a letter chain id into a number index (A->0, B->1 etc)

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
    The coordinates are located by row number (usually is atom index minus one, which is also why it's zero-based) by mdtraj but when the atom indices are not continuous there is a problem so a safer way to locate the coordinates is through row number (as a fake atom index) in case the atom indices are not continuous.
    '''
    count = 0  # keep track of row indices as "atom indices"
    for line in atoms:
        # for the specified chain
        if line[5] == chain_index:
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
            dih = np.delete(dih, (i), axis=0)
            #print(
            #    'The "' + str(dih_names[i]) +
            #    '" dihedral will not be computed due to missing coordinates.')
            dih_names.remove(dih_names[i])
            check_flag = 0
    for i in range(len(dis)):
        if 0 in dis[i]:
            dis = np.delete(dis, (i), axis=0)
            #print(
            #    'The "' + str(dis_names[i]) +
            #    '" distance will not be calculated due to missing coordinates.'
            #)
            dis_names.remove(dis_names[i])
            check_flag = 0
    if check_flag:
        pass 
        #print(
        #    "There is no missing coordinates.  All dihedrals and distances will be computed."
        #)

    '''
    # calculate the dihedrals and distances for the user-specifed structure (a static structure or an MD trajectory)
    if coord == 'pdb':
        traj = md.load(str(pdb) + '.pdb')
    dihedrals = md.compute_dihedrals(traj, dih)
    distances = md.compute_distances(traj, dis)
    #print("Key dihedrals relevant to kinase conformation are as follows:")
    #print(dih_names)
    #print(dihedrals/np.pi*180) # dihedrals in degrees
    #print(dihedrals)  # dihedrals in radians
    #print("Key distances relevant to kinase conformation are as follows:")
    #print(dis_names)
    #print(distances)
    '''
    # clean up
    rm_file = 'rm ./' + str(pdb) + '.pdb'
    subprocess.call(rm_file, shell=True)
    #del traj, dih, dis
    #del traj
    #return dihedrals, distances
    return dih, dis
def main(pdb, chain):
    numbering = basics(pdb, chain)
    (dih, dis) = features(pdb, chain, numbering)
    return dih, dis 
