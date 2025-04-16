

for e in {0..4}
do
  for f in {0..9}
  do
    python3 main.py -f $f -e $e -d guideseq -m dnabert --train 
    python3 main.py -f $f -e $e -d guideseq -m dnabert --test
    python3 main.py -f $f -e $e -d guideseq -m dnabert-epi --train --test
    python3 main.py -f $f -e $e -d guideseq -m gru-embed --train --test
    python3 main.py -f $f -e $e -d guideseq -m crispr-bert --train --test
    python3 main.py -f $f -e $e -d guideseq -m crispr-hw --train --test
    python3 main.py -f $f -e $e -d guideseq -m crispr-dipoff --train --test
    python3 main.py -f $f -e $e -d guideseq -m crispr-bert-2025 --train --test

    python3 main.py -f $f -e $e -d changeseq -m dnabert --train
    python3 main.py -f $f -e $e -d changeseq -m dnabert --test
    python3 main.py -f $f -e $e -d changeseq -m gru-embed --train --test
    python3 main.py -f $f -e $e -d changeseq -m crispr-bert --train --test
    python3 main.py -f $f -e $e -d changeseq -m crispr-hw --train --test
    python3 main.py -f $f -e $e -d changeseq -m crispr-dipoff --train --test
    python3 main.py -f $f -e $e -d changeseq -m crispr-bert-2025 --train --test

    python3 main.py -f $f -e $e -d transfer -m dnabert --train
    python3 main.py -f $f -e $e -d transfer -m dnabert --test
    python3 main.py -f $f -e $e -d transfer -m dnabert-epi --train --test
    python3 main.py -f $f -e $e -d transfer -m dnabert-epi-ablation --train --test
    python3 main.py -f $f -e $e -d transfer -m gru-embed --train --test
    python3 main.py -f $f -e $e -d transfer -m crispr-bert --train --test
    python3 main.py -f $f -e $e -d transfer -m crispr-hw --train --test
    python3 main.py -f $f -e $e -d transfer -m crispr-dipoff --train --test
    python3 main.py -f $f -e $e -d transfer -m crispr-bert-2025 --train --test
  done
done
