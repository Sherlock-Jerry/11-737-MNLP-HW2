from datasets import load_dataset, DatasetDict,concatenate_datasets
common_voice_gn = DatasetDict()
common_voice_gn["test"] = load_dataset( "mozilla-foundation/common_voice_12_0", "gn", split="test")
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition","output_dir")

common_voice_gn=common_voice_gn.select_columns(["audio", "sentence"])
print(len(common_voice_gn["test"]))
pred=[]
true=[]
for ele in common_voice_gn["test"]:
    pred.append(pipe(ele["audio"])['text'].lower().replace(',','').replace('.',''))
    true.append(ele['sentence'])

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

pred_norm=[normalizer(p) for p in pred]
true_norm=[normalizer(t) for t in true]

f=open("gn_only_pred.txt","w")
f.writelines('\n'.join(pred_norm))
f.close()

print(metric.compute(predictions=pred_norm,references=true_norm))
