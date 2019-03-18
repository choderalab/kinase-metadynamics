"""
klifs.py
Defines the Klifs class

"""

class Klifs(object):

    def __init__(self, pdb, chain, kinase_id, name, struct_id, ligand, pocket_seq, numbering):
        """This script defines a Klifs class of which any kinase can be represented as an object with the
        following parameters:

        Parameters
        ----------
        pdb: str
            The PDB code of the structure.
        chain: str
            The chain index of the structure.
        kinase_id: int
            The standard ID of a kinase enforced by the KLIFS database.
        name: str
            The standard name of the kinase used by the KLIFS database.
        struct_id: int
            The ID associated with a specific chain in the pdb structure of a kinase.
        ligand: str
            The ligand name as it appears in the pdb file.
        pocket_seq: str
            The 85 discontinuous residues (from multi-sequence alignment) that define the binding pocket of a kinase.
        numbering: list of int
            The residue indices of the 85 pocket residues specific to the structure.

        """

        self.pdb = pdb
        self.chain = chain
        self.kinase_id = kinase_id
        self.name = name
        self.struct_id = struct_id
        self.ligand = ligand
        self.pocket_seq = pocket_seq
        self.numbering = numbering
