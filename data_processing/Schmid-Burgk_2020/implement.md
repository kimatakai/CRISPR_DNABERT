
1. Docker pull

If a series of pipeline is not installed, download the environment using docker.
```bash
docker pull liyc1989/tsailabsj
```


2. Download application
```bash
git clone https://github.com/tsailabSJ/guideseq.git
```

guideseq -> database directory

3. Docker run practicically
```bash
docker run --rm -v "/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/":/data liyc1989/tsailabsj sh -c "ls -la /data"
```

4. Download reference genome
```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```

5. Reference genome indexing
```bash
docker run --rm -v "/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/":/data -v "/mnt/e/database/1_crispr_off_on_target_2025/reference":/reference liyc1989/tsailabsj sh -c "cd /; bwa index reference/hg38.fa"
```

6. Downoad Fastq files
Refer "https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRP242700&o=acc_s%3Aa"
```bash
fasterq-dump --outdir /mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913630/ --split-files SRR10913630
fasterq-dump --outdir /mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913628/ --split-files SRR10913628
fasterq-dump --outdir /mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913611/ --split-files SRR10913611
fasterq-dump --outdir /mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/SRA/SRR10913619/ --split-files SRR10913619
```

7. Preprocess fastq files
```bash
bash ./preprocess_before_mapping.sh
```

8. Mapping to reference genome
```bash
bash ./mapping.sh
```

9. Output off-target site candidates
Use SWOffinder
```bash
git clone https://github.com/OrensteinLab/SWOffinder.git
cd SWOffinder/
javac -d bin SmithWatermanOffTarget/*.java
java -cp bin SmithWatermanOffTarget.SmithWatermanOffTargetSearchAlign \
/mnt/e/database/1_crispr_off_on_target_2025/reference/hg38.fa \
/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/sgrna_list.txt \
/mnt/e/database/1_crispr_off_on_target_2025/Schmid-Burgk_2020/OTS_candidates/ \
6 6 4 1 24 true 1000 NGG true
```

10. Count Double stranded Breaks
```bash
python3 count_dsb.py
```