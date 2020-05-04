import pandas as pd
import numpy as np
import os
from Bio import AlignIO
import time

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")

pd.options.display.width = 0  # adjust according to terminal width
DEC = 3  # decimal places for rounding
NAME_DELIM = "."
LABEL_INDEX = 1  # index of name field which denotes the subtype
ID_INDEX = 2  # index of name field denoting the unique identifier

alphabet = list("ACGTN-")
base_to_int = dict(zip(alphabet, range(len(alphabet))))
int_to_base = dict(zip(range(len(alphabet)), alphabet))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_alignment", type=str, help="File containing sequence alignment")
    parser.add_argument("seq_alignment_format", type=str, help="File format for alignment, using biopython terms")
    args = parser.parse_args()
    try:
        align = AlignIO.read(args.seq_alignment, args.seq_alignment_format)
    except ValueError:  # raised if 0 or >1 alignments in file
        raise ValueError("Oops: problem reading alignment file.\n(Is the format correct?)")
    except IOError:
        raise ValueError("Oops: can't find alignment file")
    except Exception:
        raise ValueError("Oops: problem reading alignment file")

    t1 = time.time()
    # data0 = []
    # for record in align:
    #     split = record.name.split(NAME_DELIM)
    #     label, identifier = split[LABEL_INDEX], split[ID_INDEX]
    #     encoded_seq = np.array([label, identifier])
    #     seq = record.seq.upper()
    #     for state in seq:
    #         if state not in alphabet:
    #             raise ValueError(f"Sequence state {state} not recognised, in seq with name {record.name}")
    #         encoded_state = np.zeros(len(alphabet))
    #         encoded_state[base_to_int[state]] = 1
    #         encoded_seq = np.concatenate((encoded_seq, encoded_state))
    #     data0.append(encoded_seq)
    # print(data0)
    print()

    data = np.zeros((len(align), align.get_alignment_length()*len(alphabet)))
    for iRecord in range(len(align)):
        seq = align[iRecord].seq.upper()
        for iSite in range(align.get_alignment_length()):
            if not seq[iSite] in alphabet:
                raise ValueError(f"Sequence state {seq[iSite]} not recognised, in seq with name {align[iRecord].name}")
            data[iRecord, iSite*len(alphabet)+base_to_int[seq[iSite]]] = 1.0
    t2 = time.time()
    # TODO could use multiindexer instead? for states and sites
    header = [f"{str(site)}_{state}" for site in range(1, align.get_alignment_length() + 1) for state in alphabet]
    df = pd.DataFrame(data, columns=header)

    identifiers, labels = [], []
    for record in align:
        split = record.name.split(NAME_DELIM)
        labels.append(split[LABEL_INDEX])
        identifiers.append(split[ID_INDEX])
    df["label"] = labels
    df["id"] = identifiers
    print(f"time={(t2-t1)}")
    print(df)


if __name__ == "__main__":
    main()
