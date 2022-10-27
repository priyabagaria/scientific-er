# train only cls head
# python train.py \
#     --linear-ep 25 \
#     --batch-size 8 \
#     --linear-lr 1e-3 \
#     --save-models \
#     --tb 


# train only by finetuning backbone
python train.py \
    --finetune-backbone \
    --max-ep 5 \
    --batch-size 8 \
    --lr 1e-7 \
    --cls_weights ../saved_models/BERTPlus_scibert_8_25_0.001_linear_probing \
    --save-models \
    --tb 
