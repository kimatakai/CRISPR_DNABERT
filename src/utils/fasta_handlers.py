
import tqdm
import pysam
import multiprocessing
from multiprocessing import Pool, cpu_count
import os
import pandas as pd

import utils.sequence_module as sequence_module


def return_chrom_size(reference: pysam.FastaFile) -> dict:
    chrom_size = {}
    for chrom in reference.references:
        chrom_size[chrom] = reference.get_reference_length(chrom)
    chrom_size = {k: v for k, v in chrom_size.items() if k.startswith('chr') and (k[3:].isdigit() or k[3:] in ['X', 'Y'])}
    return chrom_size


def load_fasta_file(fasta_path: str) -> dict:
    # return {"seq_id": [], "seq": []}
    """
    > seq_id_1
    ATCGATCGATCGATCG
    > seq_id_2
    GCTAGCTAGCTAGCTA
    ...
    """
    fasta_data = {"seq_id": [], "seq": []}
    with open(fasta_path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                fasta_data["seq_id"].append(line[1:])
            else:
                fasta_data["seq"].append(line)
    return fasta_data


def save_flanking_sequence_fasta(
    reference_fa: pysam.FastaFile,
    off_target_data: set, # (sgRNA, chrom, strand, chromStart)
    chrom_size: dict,
    window_size: int,
    output_fasta_path: list[str, str]
    ) -> None:
    # Output list
    output_r1 = [] # objective strand
    output_r2 = [] # complementary strand
    
    for sgRNA, chrom, strand, chromStart in tqdm.tqdm(off_target_data, desc="Calculating flanking sequences...", total=len(off_target_data)):
        
        if chrom not in chrom_size:
            continue  # Skip if chromosome is not in the reference
        
        start_pos = chromStart - window_size
        end_pos = chromStart + window_size
        
        # Ensure positions are within chromosome bounds. if they are negative or exceed chromosome size, pad "N"s.
        if start_pos >= 0:
            start_pos_idx = start_pos
        else:
            start_pos_idx = 0
        if end_pos <= chrom_size[chrom]:
            end_pos_idx = end_pos
        else:
            end_pos_idx = chrom_size[chrom]
        
        # Fetch the sequence from the FASTA file
        sequence = reference_fa.fetch(chrom, start_pos_idx, end_pos_idx)
        sequence = sequence.upper()
        
        # Pad with "N"s
        if start_pos < 0:
            sequence = "N" * abs(start_pos) + sequence
            print(f"Warning: start position {start_pos} is negative, padding with 'N's.")
        if end_pos > chrom_size[chrom]:
            sequence += "N" * (end_pos - chrom_size[chrom])
            print(f"Warning: end position {end_pos} exceeds chromosome size, padding with 'N's.")
        
        # Strand handling
        if strand == "+":
            output_r1.append(f">{sgRNA}_{chrom}_{strand}_{chromStart}\n")
            output_r1.append(f"{sequence}\n")
            complementary_sequence = sequence_module.return_complementary_strand(sequence)
            output_r2.append(f">{sgRNA}_{chrom}_{strand}_{chromStart}\n")
            output_r2.append(f"{complementary_sequence}\n")
        elif strand == "-":
            complementary_sequence = sequence_module.return_complementary_strand(sequence)
            output_r1.append(f">{sgRNA}_{chrom}_{strand}_{chromStart}\n")
            output_r1.append(f"{complementary_sequence}\n")
            output_r2.append(f">{sgRNA}_{chrom}_{strand}_{chromStart}\n")
            output_r2.append(f"{sequence}\n")
        else:
            raise ValueError(f"Invalid strand value: {strand}. Expected '+' or '-'.")
    
    # Write to output files
    with open(output_fasta_path[0], 'w') as f1, open(output_fasta_path[1], 'w') as f2:
        f1.writelines(output_r1)
        f2.writelines(output_r2)
            

class FlankingSequenceProcessor:
    def __init__(self, config: dict, dataset_str: str, reference_fasta: str, chrom_size: dict):
        self.config = config
        self.dataset_str = dataset_str
        self.reference_fa = reference_fasta
        self.chrom_size = chrom_size
        self.eda_window_size = config["parameters"]["eda_window_size"]
        self.cpg_sliding_window_size = config["parameters"]["cpg_sliding_window_size"]
        self.cpg_sliding_window_step = config["parameters"]["cpg_sliding_window_step"]
        if dataset_str == "Lazzarotto_2020_CHANGE_seq" or dataset_str == "Lazzarotto_2020_GUIDE_seq":
            self.output_fasta_path_r1 = config["paths"]["fasta_file"][dataset_str]["CSGS_flanking_r1"]
            self.output_fasta_path_r2 = config["paths"]["fasta_file"][dataset_str]["CSGS_flanking_r2"]
            self.output_fasta_path_all = config["paths"]["fasta_file"][dataset_str]["CSGS_flanking_all"]
    
    @staticmethod
    def return_complementary_strand(sequence: str) -> str:
        complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
        complementary_seq = []
        for base in sequence:
            complementary_seq.append(complement_map.get(base, base))
        return "".join(complementary_seq[::-1])  # Reverse the sequence to get the complementary strand
    
    @staticmethod
    def _process_single_entry_worker(args) -> tuple | None:
        fasta_path, chrom_sizes, window_size, entry = args
        sgRNA = entry["sgRNA"]
        chrom = entry["chrom"]
        strand = entry["Align.strand"]
        chromStart = entry["Align.chromStart"]
        
        if chrom not in chrom_sizes:
            return None
        
        try:
            local_reference = pysam.FastaFile(fasta_path)
            start_pos = chromStart - window_size
            end_pos = chromStart + window_size
            fetch_start_idx = max(0, start_pos)
            fetch_end_idx = min(end_pos, chrom_sizes[chrom])
            
            sequence = local_reference.fetch(chrom, fetch_start_idx, fetch_end_idx).upper()
            if start_pos < 0:
                sequence = "N" * abs(start_pos) + sequence
            if end_pos > chrom_sizes[chrom]:
                sequence += "N" * (end_pos - chrom_sizes[chrom])
            
            header = f">{sgRNA}_{chrom}_{strand}_{chromStart}\n"
            
            if strand == "+":
                complementary_sequence = FlankingSequenceProcessor.return_complementary_strand(sequence)
                local_reference.close()
                return (header, sequence + "\n", header, complementary_sequence + "\n")
            elif strand == "-":
                complementary_sequence = FlankingSequenceProcessor.return_complementary_strand(sequence)
                local_reference.close()
                return (header, complementary_sequence + "\n", header, sequence + "\n")
            else:
                local_reference.close()
                raise ValueError(f"Invalid strand value: {strand}. Expected '+' or '-'.")
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
            return None
    
    def process_all_entries(self, dataset: pd.DataFrame, num_workers: int = 24) -> None:
        worker_args = []
        for _, entry in dataset.iterrows():
            entry_dict = {
                "sgRNA": entry["sgRNA"],
                "chrom": entry["chrom"],
                "Align.strand": entry["Align.strand"],
                "Align.chromStart": entry["Align.chromStart"]
            }
            worker_args.append((self.reference_fa, self.chrom_size, self.eda_window_size, entry_dict))
        
        output_r1_lines = []
        output_r2_lines = []
        
        if not os.path.exists(self.output_fasta_path_r1) or not os.path.exists(self.output_fasta_path_r2):
            num_workers = min(num_workers, cpu_count()-2)
            print(f"Using {num_workers} workers for processing.")
            with Pool(processes=num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(self._process_single_entry_worker, worker_args), total=len(worker_args), desc="Processing entries"))

            for result in results:
                if result is not None:
                    header_r1, seq_r1, header_r2, seq_r2 = result
                    output_r1_lines.append(header_r1)
                    output_r1_lines.append(seq_r1)
                    output_r2_lines.append(header_r2)
                    output_r2_lines.append(seq_r2)

            # Write to output files
            with open(self.output_fasta_path_r1, 'w') as f1, open(self.output_fasta_path_r2, 'w') as f2:
                f1.writelines(output_r1_lines)
                f2.writelines(output_r2_lines)