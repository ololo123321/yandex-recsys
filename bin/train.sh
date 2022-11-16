train_data_path=""
valid_data_path=""
track_to_artist_path=""
track_vocab_path=""
artist_vocab_path=""
output_dir=""

#experiment="default"
experiment="test"

rm -r ${output_dir}

python ../jobs/train.py \
  +experiment=${experiment} \
  train_data_path=${train_data_path} \
  valid_data_path=${valid_data_path} \
  track_to_artist_path=${track_to_artist_path} \
  track_vocab_path=${track_vocab_path} \
  artist_vocab_path=${artist_vocab_path} \
  hydra.run.dir=${output_dir} \
  training_args.output_dir=${output_dir}