# Create the model repository and download ResNet-50 model
MODEL_REPOSITORY=model-repository
MODEL_NAME=resnet50
SAVED_MODEL_DIR=resnet50_saved_model

mkdir -p $MODEL_REPOSITORY/$MODEL_NAME/1
cp -a $SAVED_MODEL_DIR $MODEL_REPOSITORY/$MODEL_NAME/1/

# Create Triton config
cat >$MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt <<EOL
instance_group {
 kind: KIND_CPU
}
default_model_filename: "$SAVED_MODEL_DIR"
backend: "max"
EOL

# run the recently built max_serving_local container
docker run -it --rm --network=host \
 -v $PWD/$MODEL_REPOSITORY/:/models \
 public.ecr.aws/modular/max-serving-de \
 tritonserver --model-repository=/models --model-control-mode=explicit \
 --load-model=$MODEL_NAME