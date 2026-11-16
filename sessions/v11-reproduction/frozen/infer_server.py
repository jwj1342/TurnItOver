"""One persistent frozen Ordinary 2B model serving file-queued public episodes."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor,GenerationConfig,Qwen3VLForConditionalGeneration
from turnitover.oracle.training import tokenize_prompt

def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v),encoding='utf-8');temp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('queue',type=Path);a=p.parse_args()
    a.queue.mkdir(exist_ok=True)
    basepath=a.root/'Qwen3-VL-2B-Instruct'
    processor=AutoProcessor.from_pretrained(basepath,local_files_only=True,trust_remote_code=False)
    base=Qwen3VLForConditionalGeneration.from_pretrained(basepath,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,a.root/'runs/ordinary-20261013/checkpoint',local_files_only=True).eval()
    generation=GenerationConfig.from_pretrained(basepath,local_files_only=True);generation.max_new_tokens=512;generation.do_sample=False
    write(a.queue/'ready.json',{'status':'ready','adapter':'ordinary-20261013','max_new_tokens':512,'do_sample':False})
    while not (a.queue/'stop').exists():
        requests=sorted(a.queue.glob('*.request.json'))
        if not requests:time.sleep(.1);continue
        for path in requests:
            response=path.with_name(path.name.replace('.request.json','.response.json'))
            if response.exists():continue
            try:
                job=json.loads(path.read_text());r=json.loads(Path(job['request']).read_text())
                images=[]
                for name in r['image_paths']:
                    with Image.open(name) as img:images.append(img.convert('RGB'))
                _,inputs=tokenize_prompt(processor,r['record'],images);inputs=inputs.to(model.device)
                with torch.inference_mode():generated=model.generate(**inputs,generation_config=generation)
                n=inputs['input_ids'].shape[1];suffix=generated[:,n:]
                raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                eos=generation.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                finish='stop' if int(suffix[0,-1]) in eos else 'length' if suffix.shape[1]>=512 else 'stop'
                write(response,{'status':'complete','raw':raw,'input_tokens':n,'output_tokens':suffix.shape[1],'finish_reason':finish})
            except Exception as exc:write(response,{'status':'error','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        time.sleep(.1)
    print('SERVER_STOPPED',flush=True)
if __name__=='__main__':main()
