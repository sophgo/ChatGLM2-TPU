#!/bin/bash
set -ex
models=

outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

seqlen=512
model_transform.py \
    --model_name embedding \
    --model_def ../embedding.onnx \
    --input_shapes [[$seqlen]] \
    --mlir embedding_${seqlen}.mlir


model_deploy.py \
    --mlir embedding_$seqlen.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_${seqlen}_f16.bmodel

model_transform.py \
    --model_name embedding \
    --model_def ../embedding.onnx \
    --input_shapes [[1]] \
    --mlir embedding_1.mlir


model_deploy.py \
    --mlir embedding_1.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_1_f16.bmodel

models=$models' '$outdir'/embedding_1_f16.bmodel '$outdir'/embedding_'$seqlen'_f16.bmodel '

popd

echo $models

outdir=tmp/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../lm_head.onnx \
    --mlir lm_head.mlir


model_deploy.py \
    --mlir lm_head.mlir \
    --quantize F16 \
    --quantize_table ../../lm_qtable \
    --chip bm1684x \
    --model lm_head.bmodel

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=tmp/glm_block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for i in {0..27}
do

model_transform.py \
    --model_name glm_block_$i \
    --model_def ../glm_block_$i.onnx \
    --mlir glm_block_$i.mlir


model_deploy.py \
    --mlir glm_block_$i.mlir \
    --quantize F16 \
    --quantize_table ../../glm_qtable \
    --chip bm1684x \
    --model glm_block_$i.bmodel

model_transform.py \
    --model_name glm_block_cache_$i \
    --model_def ../glm_block_cache_$i.onnx \
    --mlir glm_block_cache_$i.mlir


model_deploy.py \
    --mlir glm_block_cache_$i.mlir \
    --quantize F16 \
    --quantize_table ../../glm_qtable \
    --chip bm1684x \
    --model glm_block_cache_$i.bmodel

models=${models}${outdir}'/glm_block_'$i'.bmodel '$outdir'/glm_block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o chatglm2-6b.bmodel
