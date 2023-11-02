# Subtask 1 - Individual ASR Models

## Results
We used FairSeq to train ASR models on Guarani and Quechua languages, with the following results (WER) below
Languages | Unigram | Char | BPE
--- | --- | --- | ---
Guarani | 100.0 | 101.0 | 100.0
Quechua | 413.85 | 100.0 | 343.74

Models were trained for 30 epochs, with other hyperparameters the same as in the starter notebook
Beam width = 10 was used during generation

All the code files and processed dataset are on this [Google Drive](https://drive.google.com/drive/folders/1JiKZnVkL9Hu-Anr6jvzHyVsyBGCApLAx?usp=sharing)

## File Structure
1. `hw2_subtask1.ipynb` - The main code (modified starter notebook)
2. `data_utils.py` - Copied from [Fairseq Github](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/data_utils.py)
3. `prep_librispeech_data.py` and `preprocess_quechua_data.py` - The preprocessing scripts for processing Guarani and Quechua, inspired from Fairseq Speech to Text examples, [prep_librispeech_data.py](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/prep_librispeech_data.py)