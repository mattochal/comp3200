zip -r lolaom_ST_space.zip lolaom_ST_space/

scp -r mo4g15@iridis4_a.soton.ac.uk:~/scheduling_internship_2017/high-low-value-freq-ratio.zip ./
scp -r mo4g15@iridis4_a.soton.ac.uk:~/scheduling_internship_2017/400-experiment-final.zip ./

scp -r mo4g15@iridis4_a.soton.ac.uk:~/comp3200/lolaom_random_init_long_epochs.zip ./results/
scp -r mo4g15@iridis4_a.soton.ac.uk:~/comp3200/lolaom_long_epochs.zip ./results/
scp -r mo4g15@iridis4_a.soton.ac.uk:~/comp3200/lola_random_init_long_epochs.zip ./results/

zip lolaom_random_init_long_epochs.zip lolaom_random_init_long_epochs/
zip lolaom_long_epochs.zip lolaom_long_epochs/

unzip results/lolaom_random_init_long_epochs.zip
unzip results/lolaom_long_epochs.zip
unzip results/lola_random_init_long_epochs.zip

find . -maxdepth 1 -name "*.o5089*" -delete
find . -maxdepth 1 -name "*.e5089*" -delete

find . -maxdepth 1 -name "*.e5095577*" -printf "\n%p\n" -exec cat {} \; | tail -1

find . -maxdepth 1 -name "output_*.e497*" -printf "\n%p\n" -exec cat {} \;

find . -maxdepth 1 -name "dojo_*.o499*" -printf "\n%p\n" -exec cat {} \;

while 

cat high-low-value-freq-ratio-2/exp10/output_optimal_run

deadline-variation/exp27/output_optimal_run

Saving to deadline-variation-simplified/exp04/config.json
['qsub', 'deadline-variation-simplified/exp04/output_optimal_run']
4976604.blue101

lolaom_vs_lolaom_IPD_run.e5087521

for a in {5095760..5095987}; do find . -maxdepth 1 -name "*.$a*" -printf "\n%p\n" -exec cat {} \; done

for a in {5089286..5089333}; do echo "$a"; done | while read a; do qdel "$a"; done

for a in {4998739..4998785}; do echo "$a"; done | while read a; do find . -maxdepth 1 -name "*.$a*" -printf "\n%p\n" -exec cat {} \;; done

Saving to deadline-variation/exp27/config.json
['qsub', 'deadline-variation/exp27/output_optimal_run']
4976543.blue101
Saving to high-low-value-freq-ratio-2/exp10/config.json
['qsub', 'high-low-value-freq-ratio-2/exp10/output_optimal_run']
4976544.blue101

cat to_zip | zip -r 400-experiment-var-final -@






