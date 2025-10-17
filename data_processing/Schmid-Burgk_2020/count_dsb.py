

import config

import os
import pysam
import csv
import pandas as pd 
from collections import defaultdict



def count_dsb_for_ots_candidates(off_target, bamfile):
    chr_str = off_target["chr"]
    sgRNA_start = off_target["start_pos"]
    sgRNA_end = off_target["end_pos"]
    strand = off_target["strand"]
    
    # Calculate the center of the window of interest based on strand
    if strand == "+":
        woi_center = sgRNA_end - config.EXPECTED_CUT_DIST_FROM_SGRNA_END_PLUS_STRAND
    else:
        woi_center = sgRNA_start + config.EXPECTED_CUT_DIST_FROM_SGRNA_END_MINUS_STRAND
    
    # Define the window of interest
    woi_start = max(0, woi_center - config.WOI_PADDING)
    woi_end = woi_center + config.WOI_PADDING

    # Unique set of read start position and read end position for removing duplicates
    # tuple_set = set()
    unique_fragments = set()
    
    # Unique reads count by 20-bt window
    # eky: start position of window, value: number of unique reads in that window
    peak_counts_by_window = defaultdict(set)

    # # Fetch reads in the window of interest
    # This count is number of reads in WOI, be careful that it is not final DSB count
    raw_reads_in_woi_count = 0
    
    for read in bamfile.fetch(chr_str, woi_start, woi_end):
        raw_reads_in_woi_count += 1 # Simply counting reads in the window of interest
        
        # Check if the read is paired and not unmapped
        if not read.mate_is_unmapped and read.is_paired:
            # Calculate the span of the paired read
            span = abs(read.template_length)
            
            # Filter reads based on span
            if span >= config.MIN_FRAGMENT_SPAN and span <= config.MAX_FRAGMENT_SPAN:
                # Identify the unique fragment by its start and end positions (for deduplication)
                # fragment_start is the 5' side of the read
                fragment_start = min(read.reference_start, read.next_reference_start)
                fragment_end = fragment_start + abs(read.template_length)
                
                fragment_identifier = (chr_str, fragment_start, fragment_end)
                
                # Because of no UMI tag, reckon that the same read sequenced multiple times is a duplicate
                if fragment_identifier not in unique_fragments:
                    unique_fragments.add(fragment_identifier)
                    
                    # Add read starting position to the peak_reads_by_window
                    # Calculate read start position into which 20-bp window it falls
                    # (ex. 41225465 falls into 41225460-41225480)
                    for window_start_candidate in range(max(woi_start, fragment_start - config.PEAK_WINDOW_SIZE +1), min(woi_end, fragment_start + config.PEAK_WINDOW_SIZE)):
                        if woi_start <= window_start_candidate <= woi_end - config.PEAK_WINDOW_SIZE + 1:
                            # Increment the count of unique reads in this 20-bp window
                            peak_counts_by_window[window_start_candidate].add(fragment_identifier)
    
    final_dsb_count = 0
    
    # Evaluate peaks in the peak_counts_by_window and count DSBs
    for window_start, fragment_ids_in_window in peak_counts_by_window.items():
        # Exist enough unique reads in 20-bp window?
        if len(fragment_ids_in_window) >= config.MIN_UNIQUE_READS_FOR_PEAK:
            # Calculate the peak center
            peak_position = window_start + config.PEAK_WINDOW_SIZE // 2
            
            # Calculate cut site score
            cut_site_score = peak_position - woi_center 
            
            # Check if the cut site score is within the expected range
            if config.CUT_SITE_SCORE_RANGE[0] <= cut_site_score <= config.CUT_SITE_SCORE_RANGE[1]:
                # Consider this peank as a DSB event
                final_dsb_count += len(fragment_ids_in_window)
    
    return final_dsb_count, raw_reads_in_woi_count


def get_flanking_sequence(off_target: dict, fasta):
    start_pos = off_target["start_pos"] - 25
    end_pos = off_target["end_pos"] + 25
    
    # Fetch the sequence from the FASTA file
    sequence = fasta.fetch(off_target["chr"], max(0, start_pos), end_pos)
    sequence = sequence.upper()
    return sequence



def main():
    
    import time
    start_time = time.time()
    
    bamfile = pysam.AlignmentFile(config.BAM_file_path, "rb")
    fasta = pysam.FastaFile(config.fasta_file_path)
    
    output_data = [["sgRNA", "chrom", "SiteWindow", "Align.strand", "Align.chromStart", "Align.chromEnd", "Align.off-target", "Align.sgRNA", "Align.#Mismatches", "Align.#Bulges", "reads"]]
    metadata = [["sgRNA_name", "sgRNA_sequence", "Number_of_candidates", "Number_of_targets", "Number_of_DSBs"]]
    
    with open(config.sgrna_name_tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        sgrna_name_and_seq = {rows[0]: rows[1] for rows in reader}
    
    for sgrna_name, sgrna_seq_ in sgrna_name_and_seq.items():
        print(f"Processing sgRNA: {sgrna_name} with sequence: {sgrna_seq_}")
        sgrna_seq = sgrna_seq_ + "NGG"
        ots_candidates_csv_path = f"{config.ots_candidates_dir_path}_{sgrna_seq}.csv"
        ots_candidates_df = pd.read_csv(ots_candidates_csv_path, sep=",")
        
        n_cndidates = ots_candidates_df.shape[0]
        n_target = 0
        n_dsb = 0
        
        for index, row in ots_candidates_df.iterrows():
            chr_str = row["Chromosome"]
            strand = row["Strand"]
            end_pos = row["EndPosition"]
            align_off_target = row["AlignedText"]
            align_sgrna = row["AlignedTarget"]
            mismatch = row["#Mismatches"]
            bulge = row["#Bulges"]
            
            if "-" in align_off_target:
                start_pos = end_pos - 22
            elif "-" in align_sgrna:
                start_pos = end_pos - 24
            else:
                start_pos = end_pos - 23
            
            off_target_candidate = {
                "chr": chr_str,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "strand": strand
            }
            
            dsb_count, raw_reads_count = count_dsb_for_ots_candidates(off_target_candidate, bamfile)
            flanking_seq = get_flanking_sequence(off_target_candidate, fasta)
        
            output_data.append([
                sgrna_seq, chr_str, flanking_seq, strand, start_pos, end_pos, align_off_target, align_sgrna, mismatch, bulge, dsb_count
            ])
            
            if dsb_count > 0:
                n_target += 1
                n_dsb += dsb_count
        
        metadata.append([
            sgrna_name, sgrna_seq, n_cndidates, n_target, n_dsb
        ])
    
    # Close the BAM and FASTA files
    bamfile.close()
    fasta.close()
    
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
    
    # Write output to CSV
    output_csv_path = config.output_csv_path
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)
    
    # Write metadata to CSV
    metadata_csv_path = output_csv_path.replace(".csv", "_metadata.csv")
    with open(metadata_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metadata)


if __name__ == "__main__":
    main()