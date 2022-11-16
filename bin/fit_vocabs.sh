track_to_artist_path=""
output_dir=""
num_special_tokens=3


python ../jobs/fit_vocabs.py \
  --track_to_artist_path=${track_to_artist_path} \
  --output_dir=${output_dir} \
  --num_special_tokens=${num_special_tokens}
