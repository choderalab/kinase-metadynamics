"""
Tools for querying the KLIFS database

http://klifs.vu-compmedchem.nl/

"""

# Setup general logging (guarantee output/error message in case of interruption)
# TODO: Can we log to the terminal instead?
import logging
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)

def query_klifs_database(pdbid, chainid):
    """
    Retrieve KLIFS information from the KLIFTS database.

    Parameters
    ----------
    pdbid: str
        The PDB code of the inquiry kinase.
    chainid: str
        The chain index of the inquiry kinase.

    Returns
    -------
    klifs_info: kinomodel.models.Klifs
        Relevant KLIFS information (including kinase id, name, structure id,
        ligand name, the 85 pocket residues and their numbering) for the desired pdbid and chain.

    """
    import urllib, requests
    # absolute import (with kinomodel installed)
    #from kinomodel.models import klifs
    # JG (temperary)
    #from features import klifs
    import klifs

    # get information of the query kinase from the KLIFS database and gives values
    # of kinase_id, name and pocket_seq (numbering)
    url = "http://klifs.vu-compmedchem.nl/api/structures_pdb_list?pdb-codes=" + str(pdbid)

    # check to make to sure the search returns valid info
    # if return is empty
    if len(requests.get(url).text) == 0:
        raise ValueError("No data found in KLIFS for pdbid '{}'.".format(pdbid))
    else:
        # clean up the info from KLIFS
        clean = requests.get(url).text.replace('true', 'True').replace('false', 'False')

    # each pdb code corresponds to multiple structures
    chain_found = False
    import ast
    for structure in ast.literal_eval(clean):
        # find the specific chain
        if structure['chain'] == str(chainid):
            kinase_id = int(structure['kinase_ID'])
            name = str(structure['kinase'])
            pocket_seq = str(structure['pocket'])
            struct_id = int(structure['structure_ID'])
            # make sure the specified structure is not an apo structure
            ligand = None
            if structure['ligand'] != 0:
                ligand = str(structure['ligand'])
            chain_found = True
    if not chain_found:
        raise ValueError("No data found for chainid '{}'."
                         "Please make sure you provide a capital letter (A, B, C, ...) as a chain ID.".format(chainid))

    # Get the numbering of the 85 pocket residues
    cmd = "http://klifs.vu-compmedchem.nl/details.php?structure_id=" + str(struct_id)
    preload = urllib.request.urlopen(cmd)
    info = urllib.request.urlopen(cmd)
    for line_number, line in enumerate(info):
        line = line.decode()
        if 'pocketResidues=[' in line:
            numbering = ast.literal_eval(
                (line[line.find('=') + 1:line.find(';')]))
    # check if there is gaps/missing residues among the pocket residues.
    # If so, enforce their indices as 0 and avoid using them to compute collective variables.
    for i in range(len(numbering)):
        if numbering[i] == -1:
            #logging.info(
            #    "Warning: There is a gap/missing residue at position: " +
            #    str(i + 1) +
            #    ". Its index will be enforced as 0 and it will not be used to compute collective variables."
            #)
            numbering[i] = 0

    '''
    # print out kinase information
    logging.info("Kinase ID: " + str(kinase_id))
    logging.info("Kinase name: " + str(name))
    logging.info("Pocket residues: " + str(pocket_seq))
    logging.info("Structure ID: " + str(struct_id))
    logging.info("Ligand name: " + str(ligand))
    logging.info("Numbering of the 85 pocket residues: " + str(numbering))
    '''

    # populate a Klifs object
    klifs_info = klifs.Klifs(pdbid, chainid, kinase_id, name, struct_id, ligand,
        pocket_seq, numbering)

    return klifs_info
