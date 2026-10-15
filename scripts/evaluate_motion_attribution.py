"""GPU-only greedy inference on public motion-attribution inputs, without Gold access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from PIL import Image
from turnitover.oracle.training import tokenize_prompt


def main(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert torch.cuda.is_available(), 'Real CUDA inference required'
    args.out.mkdir(parents=True,exist_ok=False)
    processor=AutoProcessor.from_pretrained(args.adapter or args.model,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter,local_files_only=True)
    model.eval()
    rows=[json.loads(x) for x in args.inputs.read_text().splitlines()]
    with (args.out/'predictions.jsonl').open('w') as f:
        for row in rows:
            images=[Image.open(p).convert('RGB') for p in row['images']]
            assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['images']]==row['image_sha256']
            _,inputs=tokenize_prompt(processor,row,images); inputs=inputs.to(model.device)
            start=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            ids=output[:,inputs['input_ids'].shape[1]:]
            raw=processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            result={k:row[k] for k in ('id','asset_id','condition')}
            result.update(raw=raw,input_tokens=int(inputs['input_ids'].shape[1]),output_tokens=int(ids.shape[1]),
                          seconds=time.perf_counter()-start,truncated=int(ids.shape[1])==args.max_new_tokens)
            f.write(json.dumps(result)+'\n');f.flush();print(json.dumps(result),flush=True)
    manifest={'status':'complete','model':str(args.model),'adapter':str(args.adapter),'decode':'greedy',
              'max_new_tokens':args.max_new_tokens,'gpu':torch.cuda.get_device_name(),'n':len(rows),
              'inputs_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              'model_config_sha256':hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),
              'adapter_sha256':hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest() if args.adapter else None}
    (args.out/'run.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('../TurnItOver/models/Qwen3-VL-2B-Instruct'))
    p.add_argument('--adapter',type=Path);p.add_argument('--max-new-tokens',type=int,default=512)
    main(p.parse_args())
