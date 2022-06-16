import numpy as np


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from AlphaFold - PyTorch (https://github.com/Urinx/alphafold_pytorch/blob/master/feature.py):
# -------------------------------------------------------------------------------------------------------------------------------------

def extract_hmm_profile(hhm_file, sequence, asterisks_replace=0.0):
    """Extract information from an HMM file and replace asterisks. The argument hhm_file is expected to be the stringified content of the given hhm file. That is, after opening the hhm_file (i.e. open('file.hhm', 'w') as hhm_file) and calling hhm_file.read(), you should then pass these contents into this function."""
    profile_part = hhm_file.split('#')[-1]
    profile_part = profile_part.split('\n')
    whole_profile = [i.split() for i in profile_part]
    # Strip away the header and the footer
    whole_profile = whole_profile[5:-2]
    gap_profile = np.zeros((len(sequence), 10))
    aa_profile = np.zeros((len(sequence), 20))
    count_aa = 0
    count_gap = 0
    for line_values in whole_profile:
        if len(line_values) == 23:
            # The first and the last values in line_values are metadata, so skip them.
            for j, t in enumerate(line_values[2:-1]):
                # These twenty aa_profile values represent emission probabilities
                aa_profile[count_aa, j] = (2 ** (-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_aa += 1
        elif len(line_values) == 10:
            for j, t in enumerate(line_values):
                # These ten gap_profile values represent transition probabilities (first seven) and alignment diversity (last three)
                gap_profile[count_gap, j] = (2 ** (-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_gap += 1
        elif not line_values:
            pass
        else:
            raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                             'got %d' % (line_values, len(line_values)))
    hmm_profile = np.hstack([aa_profile, gap_profile])
    assert len(hmm_profile) == len(sequence)
    return hmm_profile


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Stack Overflow (https://stackoverflow.com/questions/4119070/how-to-divide-a-list-into-n-equal-parts-python):
# -------------------------------------------------------------------------------------------------------------------------------------
def slice_list(input_list, size):
    input_size = len(input_list)
    slice_size = input_size // size
    remain = input_size % size
    result = []
    iterator = iter(input_list)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(iterator.__next__())
        if remain:
            result[i].append(iterator.__next__())
            remain -= 1
    return result
