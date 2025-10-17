#!/bin/bash

REFERENCE_GENOME="/mnt/e/database/1_crispr_off_on_target_2025/reference/hg38.fa"
OUTPUT_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/mapping"


FASTQ_DIR="/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA"

R1_1="${FASTQ_DIR}/SRR10913611/R1.fastq"
R2_1="${FASTQ_DIR}/SRR10913611/R2.fastq"
R1_2="${FASTQ_DIR}/SRR10913619/R1.fastq"
R2_2="${FASTQ_DIR}/SRR10913619/R2.fastq"
R1_3="${FASTQ_DIR}/SRR10913628/R1.fastq"
R2_3="${FASTQ_DIR}/SRR10913628/R2.fastq"
R1_4="${FASTQ_DIR}/SRR10913630/R1.fastq"
R2_4="${FASTQ_DIR}/SRR10913630/R2.fastq"


FINAL_BAM="${OUTPUT_DIR}/ttiss_wt_spcas9.bam"


# bwa mem -t 24 "${REFERENCE_GENOME}" -R '@RG\tID:SRR10913611\tSM:SRR10913611\tPL:ILLUMINA\tLB:TTISS_pool_1_rep_2' "${R1_1}" "${R2_1}" | samtools view -bS - | samtools sort -o "${OUTPUT_DIR}/SRR10913611.bam"
# bwa mem -t 24 "${REFERENCE_GENOME}" -R '@RG\tID:SRR10913619\tSM:SRR10913619\tPL:ILLUMINA\tLB:TTISS_pool_2_rep_2' "${R1_2}" "${R2_2}" | samtools view -bS - | samtools sort -o "${OUTPUT_DIR}/SRR10913619.bam"
# bwa mem -t 24 "${REFERENCE_GENOME}" -R '@RG\tID:SRR10913628\tSM:SRR10913628\tPL:ILLUMINA\tLB:TTISS_pool_2_rep_1' "${R1_3}" "${R2_3}" | samtools view -bS - | samtools sort -o "${OUTPUT_DIR}/SRR10913628.bam"
# bwa mem -t 24 "${REFERENCE_GENOME}" -R '@RG\tID:SRR10913630\tSM:SRR10913630\tPL:ILLUMINA\tLB:TTISS_pool_1_rep_1' "${R1_4}" "${R2_4}" | samtools view -bS - | samtools sort -o "${OUTPUT_DIR}/SRR10913630.bam"

# Merge all BAM files into one
samtools merge -f "${FINAL_BAM}" "${OUTPUT_DIR}/SRR10913611.bam" "${OUTPUT_DIR}/SRR10913619.bam" "${OUTPUT_DIR}/SRR10913628.bam" "${OUTPUT_DIR}/SRR10913630.bam"

samtools index "${FINAL_BAM}"