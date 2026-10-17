"""Greedy held-object evaluation and private scoring for motion diagnosis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from turnitover.oracle.motion_diagnosis import CLASSES, load_motion_examples, validate_target
from turnitover.oracle.training import tokenize_prompt
from turnitover.verifier.contracts import parse_json


def main(args):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if args.out.exists() or not torch.cuda.is_available(): raise ValueError("Fresh output and CUDA required")
    args.out.mkdir(parents=True); examples=list(load_motion_examples(args.dataset,"validation","evidence"))
    processor=AutoProcessor.from_pretrained(args.adapter,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda()
    model=PeftModel.from_pretrained(model,args.adapter,local_files_only=True); model.eval()
    confusion=defaultdict(Counter); counts=Counter(n=len(examples)); per_object=defaultdict(Counter)
    with (args.out/"predictions.jsonl").open("w") as stream:
        for record,images,source in examples:
            _,inputs=tokenize_prompt(processor,record,images); inputs=inputs.to(model.device); start=time.perf_counter()
            with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            ids=output[:,inputs["input_ids"].shape[1]:]
            raw=processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            pred={}; error=None
            try:
                pred=parse_json(raw)
                validate_target({k:v for k,v in pred.items() if k != "comparison"})
            except (ValueError,KeyError,TypeError) as exc: error=str(exc); pred={}
            gold=source["evidence_target"]; expected=gold["defect_type"]; predicted=pred.get("defect_type","invalid")
            confusion[expected][predicted]+=1; obj=per_object[source["asset_id"]]; obj["n"]+=1
            valid=predicted in CLASSES; class_ok=predicted==expected
            joint_ok=pred.get("joint")==gold["joint"]; part_ok=pred.get("part")==gold["part"]
            steps=pred.get("evidence_steps",[]); allowed=set(gold["evidence_steps"])
            evidence_ok=(predicted == "uncertain" and steps == []) or (
                predicted == "clean" and bool(steps) and set(steps) == allowed) or (
                predicted in ("axis","offset") and bool(steps) and set(steps)<=allowed)
            if "comparison" in source.get("gold", {}):
                # Axis support must include a rest/actuated pair, not just any changed image.
                pairs=source["gold"]["comparison"]["motion_pairs"]
                if predicted=="axis":
                    evidence_ok=any(set(pair)<=set(steps) for pair in pairs)
                elif predicted=="offset":
                    evidence_ok=bool(steps) and set(steps)<=set(source["gold"]["comparison"]["static_steps"])
            public=json.loads(source['messages'][0]['content'].split('Public input: ',1)[1])
            legal_steps={s['step'] for s in public['states']}
            evidence_ok=evidence_ok and all(type(s) is int for s in steps) and len(set(steps))==len(steps) and set(steps)<=legal_steps
            full=valid and class_ok and joint_ok and part_ok and evidence_ok
            comparison=pred.get("comparison",{})
            expected_comparison=source.get("gold",{}).get("comparison",{})
            comparison_scores={k:comparison.get(k)==v for k,v in expected_comparison.items()}
            for key,value in comparison_scores.items():
                counts["comparison_"+key]+=int(value); obj["comparison_"+key]+=int(value)
            counts["unsupported"]+=int(predicted in ("axis","offset") and not full)
            for key,value in (("valid",valid),("class_correct",class_ok),("joint_correct",joint_ok),
                              ("part_correct",part_ok),("evidence_correct",evidence_ok),("full_correct",full)):
                counts[key]+=int(value); obj[key]+=int(value)
            row={"id":record["id"],"asset_id":record["asset_id"],"condition":record["condition"],
                 "expected":gold,"predicted":pred or None,"valid":valid,"class_correct":class_ok,
                 "joint_correct":joint_ok,"part_correct":part_ok,"evidence_correct":evidence_ok,
                 "full_correct":full,"comparison_scores":comparison_scores,"expected_comparison":expected_comparison,"error":error,"raw":raw,"input_tokens":int(inputs["input_ids"].shape[1]),
                 "output_tokens":int(ids.shape[1]),"seconds":time.perf_counter()-start}
            stream.write(json.dumps(row)+"\n");stream.flush();print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
    full_rows=sum(1 for _,_,s in examples if s["condition"].endswith("full_motion"))
    metrics={"status":"complete","n":counts["n"],"full_motion_n":full_rows,
      **{k:counts[k]/counts["n"] for k in ("valid","class_correct","joint_correct","part_correct","evidence_correct","full_correct")},
      "comparison_correct":{k:counts["comparison_"+k] for k in ("static_rest","static_steps","motion_change","motion_pairs")},
      "unsupported":counts["unsupported"],
      "confusion":{g:{p:confusion[g][p] for p in (*CLASSES,"invalid")} for g in CLASSES},
      "per_object":{a:dict(v) for a,v in sorted(per_object.items())},"adapter":str(args.adapter),
      "adapter_sha256":hashlib.sha256((args.adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
      "dataset_manifest_sha256":hashlib.sha256((args.dataset/"manifest.json").read_bytes()).hexdigest()}
    (args.out/"metrics.json").write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--max-new-tokens",type=int,default=256)
    a=p.parse_args();a.dataset=a.dataset.resolve();a.model=a.model.resolve();a.adapter=a.adapter.resolve();a.out=a.out.resolve();main(a)
