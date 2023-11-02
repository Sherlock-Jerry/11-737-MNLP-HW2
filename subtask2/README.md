To ensure the right packages are installed to run the code in this subtask, the users can run pip install -r requirements.txt


finetune_Guarani.py - Finetunes the openai whisper model on Guarani and Spanish datasets

finetune_Guarani_only.py - Finetunes the openai whisper model on Guarani dataset only

finetune_Quechua.py - Finetunes the openai whisper model on Quechua and Swahili datasets

infer.py - Infers from the pretrained model
    
infer.py file can be used to infer for both Quechua and Guarani, by changing the language id while loading the test dataset and specifying the 
corresponding directory of the pretrained model while loading from pretrained model in line 6

The results of the experiments are noted in the following files :
1.Finetuning with Guarani dataset only[Test set CommonVoice 11.0 gn] - Guarani_only_pred.txt
2.Finetuning with Guarani and Spanish datasets[Test set CommonVoice 11.0 gn]  - Guarani_pred.txt
3.Finetuning with Quechua and Swahili datasets[Test set CommonVoice 11.0 quy] - Quechua_pred.txt

The WER results for the abve experiments are noted in the report.
