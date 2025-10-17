
from Bio import pairwise2

ATGC = ["A", "T", "G", "C", "-"]


def return_complementary_strand(sequence: str) -> str:
    complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    complementary_seq = []
    for base in sequence:
        complementary_seq.append(complement_map.get(base, base))  # Use base as is if not in complement_map
    return "".join(complementary_seq[::-1])  # Reverse the sequence to get the complementary strand


def alignment_pair_seqs(seq_rna: str, seq_dna: str, maxlen: int=24) -> tuple:
    if len(seq_rna) == len(seq_dna):
        raise ValueError("RNA and DNA sequences must be of different lengths for alignment.")
    else:
        # Perform pairwise alignment
        alignments = pairwise2.align.globalms(seq_rna, seq_dna, 1, -1, -1, -0.5) # Match, mismatch, gap open, gap extend
        seq_rna_aligned, seq_dna_aligned = alignments[0].seqA, alignments[0].seqB
        if len(seq_rna_aligned) == len(seq_dna_aligned):
            if len(seq_rna_aligned) == maxlen:
                padded_seq_rna = seq_rna_aligned
                padded_seq_dna = seq_dna_aligned
            elif len(seq_rna_aligned) < maxlen:
                padded_seq_rna = "-" * (maxlen - len(seq_rna_aligned)) + seq_rna_aligned
                padded_seq_dna = "-" * (maxlen - len(seq_dna_aligned)) + seq_dna_aligned
            else:
                raise ValueError("Aligned sequences are longer than maxlen bases. Please check the input sequences.")
        else:
            raise ValueError("Aligned sequences do not match. Please check the input sequences.")
    return padded_seq_rna, padded_seq_dna
        

def padding_hyphen_to_seq(seq_rna: str, seq_dna: str, maxlen: int=24) -> tuple:
    seq_rna_len = len(seq_rna)
    seq_dna_len = len(seq_dna)
    if seq_rna_len == seq_dna_len:  # if both sequences are of the same length
        if seq_rna_len == maxlen:
            padded_seq_rna = seq_rna
            padded_seq_dna = seq_dna
        else:
            padded_seq_rna = "-" * (maxlen - seq_rna_len) + seq_rna 
            padded_seq_dna = "-" * (maxlen - seq_dna_len) + seq_dna
    else:   # if the lengths are different
        padded_seq_rna, padded_seq_dna = alignment_pair_seqs(seq_rna, seq_dna, maxlen=maxlen)
        if seq_rna_len == maxlen:
            padded_seq_rna = seq_rna
        else:
            padded_seq_rna = "-" * (maxlen - len(padded_seq_rna)) + padded_seq_rna
        if seq_dna_len == maxlen:
            padded_seq_dna = seq_dna
        else:
            padded_seq_dna = "-" * (maxlen - len(padded_seq_dna)) + padded_seq_dna
    # Replace "N" to other pair characters
    padded_seq_rna = list(padded_seq_rna)
    padded_seq_dna = list(padded_seq_dna)
    for i, (rna_base, dna_base) in enumerate(zip(padded_seq_rna, padded_seq_dna)):
        if rna_base == "N" and dna_base in ATGC:
            padded_seq_rna[i] = dna_base
        elif dna_base == "N" and rna_base in ATGC:
            padded_seq_dna[i] = rna_base
    padded_seq_rna = "".join(padded_seq_rna)
    padded_seq_dna = "".join(padded_seq_dna)
    return padded_seq_rna, padded_seq_dna

def complete_bulge_seq(seq_rna: str, seq_dna: str) -> tuple:
    completed_seq_rna = []
    completed_seq_dna = []
    mismatch_pair = []
    for i, (r1_base, r2_base) in enumerate(zip(seq_rna, seq_dna)):
        if r1_base == r2_base:
            mismatch_pair.append(0)
        else:
            mismatch_pair.append(1)
        if r1_base in ATGC and r2_base in ATGC:
            completed_seq_rna.append(r1_base)
            completed_seq_dna.append(r2_base)
        elif r1_base in ["-", "N"] and r2_base in ATGC:
            completed_seq_rna.append(r2_base)
            completed_seq_dna.append(r2_base)
        elif r1_base in ATGC and r2_base in ["-", "N"]:
            completed_seq_rna.append(r1_base)
            completed_seq_dna.append(r1_base)
        elif r1_base in ["-", "N"] and r2_base in ["-", "N"]:
            completed_seq_rna.append("-")
            completed_seq_dna.append("-")
        else:
            raise ValueError(f"Invalid base pair at position {i}: RNA base '{r1_base}', DNA base '{r2_base}'. Expected bases are A, T, G, C, or '-' for gaps.")
    seq_rna = "".join(completed_seq_rna)
    seq_dna = "".join(completed_seq_dna)
    return seq_rna, seq_dna, mismatch_pair
            
        
    
    
    
    
    
    
    seq_rna = seq_rna.replace("-", "N")
    seq_dna = seq_dna.replace("-", "N")
    
    if len(seq_rna) != len(seq_dna):
        raise ValueError("RNA and DNA sequences must be of the same length after replacing '-' with 'N'.")
    
    return seq_rna, seq_dna