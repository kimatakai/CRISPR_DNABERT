
BASE_DIR = "/mnt/e/database/1_crispr_off_on_target_2025/"

BAM_file_path = BASE_DIR + "Schmid-Burgk_2020/mapping/ttiss_wt_spcas9.bam"
fasta_file_path = BASE_DIR + "reference/hg38.fa"
sgrna_name_tsv_path = BASE_DIR + "Schmid-Burgk_2020/sgrna_name.tsv"
ots_candidates_dir_path = BASE_DIR + "Schmid-Burgk_2020/OTS_candidates/"
output_csv_path = BASE_DIR + "Schmid-Burgk_2020/SchmidBurgk_2020_TTISS.csv"



# Parameters for counting reads
EXPECTED_CUT_DIST_FROM_SGRNA_END_PLUS_STRAND = 3
EXPECTED_CUT_DIST_FROM_SGRNA_END_MINUS_STRAND = 20
CUT_SITE_SCORE_RANGE = (-3, 3)
PEAK_WINDOW_SIZE = 20
MIN_UNIQUE_READS_FOR_PEAK = 2
MAX_FRAGMENT_SPAN = 1000
MIN_FRAGMENT_SPAN = 37
WOI_PADDING = 30

