"""
Driver for featurizing kinase models.

"""

def _parse_arguments(**kwargs):
    """Checks whether the user has provided any arguments directly to
    the main function through the kwargs keyword, and if not, attempt
    to find them from the command line arguments using argparse.

    Notes
    -----
    This method requires the pdb, chain, feature and coord args to be either
    all present in the kwargs dictionary, or given on the command line.

    Parameters
    ----------
    kwargs: dict of str, values, optional
        A dictionary of passed arguments (which may be empty), where the key is
        the name of the argument, and the value is the argument value.

    Returns
    -------
    argparse.Namespace

        A namespace containing the arguments which either were directly passed
        in as kwargs, or this which were taken as input from the comman line.

        e.g. Namespace(chain='A', coord='pdb', feature='conf', pdb='3PP0')
    """

    arguments = None

    import argparse

    if not kwargs:

        # parsing command-line arguments
        parser = argparse.ArgumentParser(
            prog='kinomodel',
            description='Featurize kinase structures',
        )
        parser.add_argument(
            '--pdb',
            required=True,
            action='store',
            nargs='?',
            type=str,
            help='the PDB code of the structure to featurize')
        parser.add_argument(
            '--chain',
            required=True,
            action='store',
            nargs='?',
            type=str,
            help='the chain index of the structure to featurize')
        parser.add_argument(
            '--feature',
            required=True,
            action='store',
            choices=['conf', 'interact', 'both'],
            nargs='?',
            help='compute collective variables related to protein conformation, protein-ligand interaction, or both'
        )
        parser.add_argument(
            '--coord',
            required=True,
            action='store',
            choices=['pdb', 'dcd'],
            nargs='?',
            type=str,
            help='the coordinates (a pdb file) or trajectories (e.g. a dcd file with associated topology info) to '
                 'featurize. Default is the pdb coordinates under the given PDB code.'
        )

        arguments = parser.parse_args()

    else:

        # Check to make sure that all of the required args are present
        # in the kwargs dictionary.

        # You might want to add your own validation logic here.
        assert 'pdb' in kwargs
        assert 'chain' in kwargs
        assert 'feature' in kwargs
        assert 'coord' in kwargs

        arguments = argparse.Namespace(**kwargs)

    return arguments

# main function
def featurize(**kwargs):
    """
    Compute structural and/or interaction features for a given PDB structure.

    Parameters
    ----------
    args: a Namespace object from argparse
        Information from parsing the command line
        e.g. Namespace(chain='A', coord='pdb', feature='conf', pdb='3PP0')

    Returns
    -------
    key_res : list of int
        Key residue indices that are involved in featurization.
    dihedrals: list of floats
        A list (one frame) or lists (multiple frames) of dihedrals relevant to kinase conformation.
    distances: list of floats
        A list (one frame) or lists (multiple frames) of intramolecular distances relevant to kinase conformation.
    AND/OR

    mean_dist: float
            A float (one frame) or a list of floats (multiple frames), which is the mean pairwise distance
            between ligand heavy atoms and the CAs of the 85 pocket residues.

    .. todo ::

       Refactor this into a featurization driver driven by documented kwargs.
       Think about the API from the perspective of someone wanting to assemble a program using this API.

    """

    # absolute import (with kinomodel installed)
    #from kinomodel.features import query_klifs
    #from kinomodel.features import protein as pf
    #from kinomodel.features import interactions as inf

    # JG (temperary)
    #from features import query_klifs
    #from features import protein as pf
    #from features import interactions as inf

    import query_klifs
    import protein as pf
    
    args = _parse_arguments(**kwargs)

    my_kinase = None

    if args.feature == "conf":
        klifs = query_klifs.query_klifs_database(args.pdb, args.chain)
        key_res = pf.key_klifs_residues(klifs.numbering)
        (dihedrals, distances) = pf.compute_simple_protein_features(args.pdb, args.chain, args.coord, klifs.numbering)
        return key_res, dihedrals, distances

    elif args.feature == "interact":
        klifs = query_klifs.query_klifs_database(args.pdb, args.chain)
        mean_dist = inf.compute_simple_interaction_features(args.pdb, args.chain, args.coord, klifs.ligand, klifs.numbering)
        return mean_dist

    elif args.feature == "both":
        klifs = query_klifs.query_klifs_database(args.pdb, args.chain)
        key_res = pf.key_klifs_residues(klifs.numbering)
        (dihedrals, distances) = pf.compute_simple_protein_features(args.pdb, args.chain, args.coord, klifs.numbering)
        mean_dist = inf.features(args.pdb, args.chain, args.coord, klifs.ligand, klifs.numbering)
        return key_res, dihedrals, distances, mean_dist
    else:
        raise Exception("Unknown feature '{}'".format(args.feature))
        return
