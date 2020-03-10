# bertogre

Use exactly like you would with original run_squad.py except: \
 \
Additional model types: \
    "bfqa" - custom Bert with layer duplication support \
    "afqa" - custom Albert with layer duplication support \
    "rfqa" - custom Roberta with layer duplication support \
    "dfqa" - custom DistilBert with layer duplication support \
 \
Additional arguments: \
    --checkpoint_prefix [name] - (optional) adds a prefix to the checkpoint folders \
    --albert_add [n] - (optional, afqa only) adds n layers to Albert before training \
    --bert_dup [n] - (optional, bfqa, rfqa, dfqa only) duplicates layer n of a Bert model with the new layer next to the original before training \
