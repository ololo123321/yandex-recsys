experiment="[\"inference/ensemble\",\"inference/final\",\"inference/valid\"]"

parent_dir="$(cd .. && pwd)"
input_path="/data/test"

docker run \
  -it \
  --rm \
  -v "${parent_dir}/models":"/models" \
  -v "${parent_dir}/data":"/data" \
  -v "${parent_dir}/outputs":"/outputs" \
  -v "${parent_dir}":"/app" \
  ololo123321/yandex-recsys:cuda11.3.0-cudnn8-devel-ubuntu20.04 python3 /app/jobs/predict.py \
    +experiment=${experiment} \
    input_path=${input_path}