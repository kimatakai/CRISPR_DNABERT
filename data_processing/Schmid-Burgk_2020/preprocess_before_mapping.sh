#!/bin/bash
# This script preprocesses the input data before mapping.
# Use fastp and seqtk to filter and process the reads.

# --- Set up environment ---
# Input files

# SRR10913630
# R1_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630/SRR10913630_1.fastq"
# R2_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630/SRR10913630_2.fastq"
# R1_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630/input_R1.fastq"
# R2_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630/input_R2.fastq"
# # Output filenames for processed FASTQs (will be gzipped)
# R1_FINAL_OUT="R1.fastq"
# R2_FINAL_OUT="R2.fastq"
# # Output files
# WORKING_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630"

# SRR10913628
# R1_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628/SRR10913628_1.fastq"
# R2_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628/SRR10913628_2.fastq"
# R1_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628/input_R1.fastq"
# R2_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628/input_R2.fastq"
# # Output filenames for processed FASTQs (will be gzipped)
# R1_FINAL_OUT="R1.fastq"
# R2_FINAL_OUT="R2.fastq"
# # Output files
# WORKING_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628"

# SRR10913611
# R1_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611/SRR10913611_1.fastq"
# R2_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611/SRR10913611_2.fastq"
# R1_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611/input_R1.fastq"
# R2_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611/input_R2.fastq"
# # Output filenames for processed FASTQs (will be gzipped)
# R1_FINAL_OUT="R1.fastq"
# R2_FINAL_OUT="R2.fastq"
# # Output files
# WORKING_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611"

# SRR10913619
R1_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619/SRR10913619_1.fastq"
R2_IN_RAW="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619/SRR10913619_2.fastq"
R1_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619/input_R1.fastq"
R2_IN="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619/input_R2.fastq"
# Output filenames for processed FASTQs (will be gzipped)
R1_FINAL_OUT="R1.fastq"
R2_FINAL_OUT="R2.fastq"
# Output files
WORKING_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619"


# Read filter patern : NNNNNNNNNNNNNNNNNNNNNNNAAC
# Read 1: Extract the length of 25 bp from the 27th base
# Read 2: Extract the length of 15 bp from the 11th base
# Check if read 1 contains AAC at the position of 24~26th base
# Firstly quality filter the reads and physically trimming
# NNNNNNNNNNNNNNNNNNNNNNNAAC (23 Ns, then AAC) -> AAC at 24, 25, 26 (1-based)
PATTERN="AAC"
PATTERN_START_1BASED=24

# Read 1 mapping start index and length
R1_TRIM_START_1BASED=26 # 1-based start for mapping (skips first 25bp)
R1_TRIM_LENGTH=25       # Length of sequence to keep for mapping
# Read 2 mapping start index and length
R2_TRIM_START_1BASED=11 # 1-based start for mapping (skips first 10bp)
R2_TRIM_LENGTH=15       # Length of sequence to keep for mapping

# ID row of Fastq files space -> underbar
# The ID row of Fastq files is space-separated, so we need to handle it accordingly.
cat "${R1_IN_RAW}" | sed 's/ /_/g' | sed 's/_length=[0-9]*//g' > "${R1_IN}"
cat "${R2_IN_RAW}" | sed 's/ /_/g' | sed 's/_length=[0-9]*//g' > "${R2_IN}"
# cat "${R2_IN_RAW}" | sed 's/ /_/g' > "${R2_IN%.temp}"
# # Remove the part of "_length=[number]" from the ID row
# cat "${R1_IN%.temp}" | sed 's/_length=[0-9]*//g' > "${R1_IN}"
# cat "${R2_IN%.temp}" | sed 's/_length=[0-9]*//g' > "${R2_IN}"

# --- Start processing ---
# Step 1: Run fastp to filter and trim reads
# -f, -t : Set the start and length of the reads to keep
# -i: リード1の入力fastqファイルを指定、-o: リード1の処理済み出力fastqファイル
# -I: リード2の入力fastqファイルを指定、-O: リード2の処理済み出力fastqファイル
# -q: Set the minimum quality score for filtering
# --average_qual: Set the average quality score for filtering
# --length_required: Set the minimum length of reads to keep
# --json: Output JSON report
# --html: Output HTML report
# if [ $? -ne 0 ]; then echo "Error in fastp. Exiting."; exit 1; fi : Check if fastp command was successful
~/bioinfomatics-tools/fastp -i "${R1_IN}" -o "${WORKING_DIR}/temp_r1_qual_trim.fastq.gz" \
      -I "${R2_IN}" -O "${WORKING_DIR}/temp_r2_qual_trim.fastq.gz" \
      -q 15 --average_qual 15 --length_required 15 \
      --json "${WORKING_DIR}/fastp.json" --html "${WORKING_DIR}/fastp.html"
if [ $? -ne 0 ]; then echo "Error in fastp. Exiting."; exit 1; fi
# gzip -kc "${R1_IN}" > "${WORKING_DIR}/temp_r1_qual_trim.fastq.gz"
# gzip -kc "${R2_IN}" > "${WORKING_DIR}/temp_r2_qual_trim.fastq.gz"
echo "Quality trimming completed."

# Step 3: Apply read filter
# Check if read 1 contains AAC at the position of 24~26th base
# -c extract read ID
# -q need because of handling of row of quality score
# 3. Apply Read Filter (NNN...AAC) based on Read 1 and synchronize paired-end reads
# This involves:
#   a. Extracting sequences from temporary R1 file.
#   b. Checking for PATTERN at PATTERN_START_1BASED.
#   c. Storing IDs of reads that pass the pattern filter.
#   d. Using seqtk subseq to extract passing reads for both R1 and R2.
echo "2. Applying Read Filter (${PATTERN} at 1-based pos ${PATTERN_START_1BASED}) and synchronizing paired-end reads..."

# Extract Read IDs for R1 sequences that match the pattern
# zcat to decompress, then awk to check pattern and print sequence identifier (header line)
# substr($1, PATTERN_START_1BASED, 3) checks 3 bases starting from PATTERN_START_1BASED
# zcat: gzip で圧縮されたファイルを解凍し、その内容を標準出力
# FastQファイルは4行で1つのリード情報を構成
cat "${R1_IN}" | \
  awk '
  BEGIN {
    FS="\n"; # Field separator is newline within a record
    RS="@";  # Record separator is '@' (start of a new FASTQ record)
  }
  NR > 1 { # Skip the first empty record created by RS="@"
    id_line = $1;      # The first field is the ID line (e.g., "READ_ID description")
    seq_line = $2;     # The second field is the sequence line
    # qual_line = $4;  # The fourth field is the quality score line (not used for pattern matching here)

    # Check the pattern in the sequence line. substr is 1-based in awk.
    # Pattern is AAC (length 3), starting at PATTERN_START_1BASED (e.g., 24)
    if (substr(seq_line, '"${PATTERN_START_1BASED}"', length("'"${PATTERN}"'")) == "'"${PATTERN}"'") {
      # Print the ID. It might contain spaces, so print the whole $1 (which is the ID line).
      # Remove any trailing newline from id_line if it was present before RS=@
      # We need to remove the first char '@' from the ID line in the previous script.
      # Here, $1 is already the ID part without the leading '@'.
      print id_line;
    }
  }' > "${WORKING_DIR}/passed_read_ids.txt"

if [ $? -ne 0 ]; then echo "Error during pattern filtering. Exiting."; exit 1; fi
echo "Pattern filtering completed. Passed IDs saved to ${WORKING_DIR}/passed_read_ids.txt."

# Filter both R1 and R2 FASTQ files using the list of passed Read IDs
# seqtk subseq filters reads based on a list of read IDs.
~/bioinfomatics-tools/seqtk/seqtk subseq "${WORKING_DIR}/temp_r1_qual_trim.fastq.gz" "${WORKING_DIR}/passed_read_ids.txt" > "${WORKING_DIR}/temp_r1_pattern_filtered.fastq"
~/bioinfomatics-tools/seqtk/seqtk subseq "${WORKING_DIR}/temp_r2_qual_trim.fastq.gz" "${WORKING_DIR}/passed_read_ids.txt" > "${WORKING_DIR}/temp_r2_pattern_filtered.fastq"
if [ $? -ne 0 ]; then echo "Error filtering reads by ID. Exiting."; exit 1; fi
echo "Paired-end reads synchronized."

# 4. Apply physical trimming to retain only mapping-relevant regions
# seqtk trimfq: trimfq <file.fq> <start_pos_1based> <end_pos_1based>
# R1: 59bp read. Want 25bp starting from 26th bp (1-based). So, keep 26 to 50.
# R2: 25bp read. Want 15bp starting from 11th bp (1-based). So, keep 11 to 25.
echo "3. Applying physical trimming..."

# R1 trimming
~/bioinfomatics-tools/seqtk/seqtk trimfq "${WORKING_DIR}/temp_r1_pattern_filtered.fastq" "${R1_TRIM_START_1BASED}" "$((R1_TRIM_START_1BASED + R1_TRIM_LENGTH - 1))" > "${WORKING_DIR}/${R1_FINAL_OUT%.gz}"
if [ $? -ne 0 ]; then echo "Error during R1 trimming. Exiting."; exit 1; fi

# R2 trimming
~/bioinfomatics-tools/seqtk/seqtk trimfq "${WORKING_DIR}/temp_r2_pattern_filtered.fastq" "${R2_TRIM_START_1BASED}" "$((R2_TRIM_START_1BASED + R2_TRIM_LENGTH - 1))" > "${WORKING_DIR}/${R2_FINAL_OUT%.gz}"
if [ $? -ne 0 ]; then echo "Error during R2 trimming. Exiting."; exit 1; fi
echo "Physical trimming completed."

# 5. Compress final output FASTQ files
# echo "4. Compressing final FASTQ files..."
# gzip "${WORKING_DIR}/${R1_FINAL_OUT%.gz}"
# gzip "${WORKING_DIR}/${R2_FINAL_OUT%.gz}"
# echo "Compression completed."

# 6. Clean up temporary files
echo "5. Cleaning up temporary files..."
rm "${WORKING_DIR}/temp_r1_qual_trim.fastq.gz" "${WORKING_DIR}/temp_r2_qual_trim.fastq.gz" \
   "${WORKING_DIR}/temp_r1_pattern_filtered.fastq" "${WORKING_DIR}/temp_r2_pattern_filtered.fastq" \
   "${WORKING_DIR}/passed_read_ids.txt"
rm "${WORKING_DIR}/fastp.json" "${WORKING_DIR}/fastp.html"
rm "${R1_IN}" "${R2_IN}" # Remove the temporary input files
echo "Cleanup completed."