import pandas as pd
import numpy as np
from Bio import AlignIO
import sys

NAME_DELIM = "."
LABEL_INDEX = 1  # index of name field which denotes the subtype
ID_INDEX = 2  # index of name field denoting the unique identifier

alphabet = list("ACGTN-")
base_to_int = dict(zip(alphabet, range(len(alphabet))))
int_to_base = dict(zip(range(len(alphabet)), alphabet))


def preproc_file(align_path, align_format):
    try:
        align = AlignIO.read(align_path, align_format)
    except ValueError:  # raised if 0 or >1 alignments in file
        raise ValueError("Oops: problem reading alignment file.\n(Is the format correct?)")
    except IOError:
        raise ValueError("Oops: can't find alignment file")
    except Exception:
        raise ValueError("Oops: problem reading alignment file")

    return preproc_align(align)


def preproc_align(align, id_col="id", ground_truth_col="label"):
    data = np.zeros((len(align), align.get_alignment_length() * len(alphabet)))
    for iRecord in range(len(align)):
        seq = align[iRecord].seq.upper()
        for iSite in range(align.get_alignment_length()):
            if not seq[iSite] in alphabet:
                raise ValueError(f"Sequence state {seq[iSite]} not recognised, in seq with name {align[iRecord].name}")
            data[iRecord, iSite * len(alphabet) + base_to_int[seq[iSite]]] = 1.0

    header = [f"{str(site)}_{state}" for site in range(1, align.get_alignment_length() + 1) for state in alphabet]
    df = pd.DataFrame(data, columns=header)

    identifiers, labels = [], []
    for record in align:
        split = record.name.split(NAME_DELIM)
        labels.append(split[LABEL_INDEX])
        identifiers.append(split[ID_INDEX])
    df[ground_truth_col] = labels
    df[id_col] = identifiers
    df = df[[id_col, ground_truth_col] + list(df.columns[:-2].values)]
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_alignment", type=str, help="File containing sequence alignment")
    parser.add_argument("seq_alignment_format", type=str, help="File format for alignment, using biopython terms")
    args = parser.parse_args()

    df = preproc_file(args.seq_alignment, args.seq_alignment_format)
    df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
