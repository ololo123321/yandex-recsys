test_data_path=""
predictions_path=""
k=100

python ../jobs/evaluate.py \
  --test_data_path=${test_data_path} \
  --predictions_path=${predictions_path} \
  --k=${k}