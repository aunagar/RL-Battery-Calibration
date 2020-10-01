bsub -n 1 -W 120:00 -R "rusage[mem=32000,ngpus_excl_p=1]" python main.py
