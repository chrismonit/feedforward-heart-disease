import pandas as pd
import os
from Bio import AlignIO

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")

pd.options.display.width = 0  # adjust according to terminal width
DEC = 3  # decimal places for rounding


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


if __name__ == "__main__":
    main()
