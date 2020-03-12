# bertogre

Use exactly like you would with original run_squad.py except: \
 \
Additional model types (smaller/base models shown to have better results for Bert, unknown for Albert): \
    "bfqa" - custom Bert with layer duplication support, intended for use with bert-base-uncased and other base models \
    "afqa" - custom Albert with layer duplication support, intended for use with albert-base-v2 \
    "rfqa" - custom Roberta with layer duplication support, intended for use with roberta-base and distilroberta-base \
    "dfqa" - custom DistilBert with layer duplication support, intended for use with distilbert-base-uncased \
    "xlnfqa" - custom XLNet with layer duplication support, intended for use with xlnet-base-cased \
    "xlmfqa" - custom XLM with layer duplication support, intended for use with xlm-mlm-en-2048 \ 
 \
Additional arguments: \
    --model_name [name] - (optional) adds a prefix to the checkpoint folders and logs, not to be confused with model_name_or_path \
    --albert_add [n] - (optional, afqa only) adds n layers to Albert before training \
    --albert_set [n] - (optional, afqa only) sets Albert to have n layers before training \
    --bert_dup [n] - (optional, non-afqa only) duplicates layer n of a Bert model with the new layer next to the original before training \
    --bert_dup_n [n] - (optional, non-afqa only) duplicates first n of a Bert model with the new layers next to the originals before training 
    --project_dir [path] - (optional) outputs submissions csv and log file to a different folder
   
   
Not tested on XLM. XLNet implementation in the original run_squad.py, and thus our version, is suspected to be faulty due to incorrect preprocessing, as per https://github.com/huggingface/transformers/issues/947.
