input_path=""
output_path=""
limit=-1

experiment="inference/joint"

python ../jobs/predict.py \
  +experiment=${experiment} \
  input_path=${input_path} \
  output_path=${output_path} \
  ++limit=${limit}