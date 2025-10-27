# Activate conda environment:
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
  echo "Running on Perlmutter"
else
  echo "Running from" $HOSTNAME
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate FIDASIM_env