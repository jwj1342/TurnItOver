"use strict";
(() => {
  const reviewer = document.documentElement.dataset.reviewer;
  const cases = JSON.parse(document.getElementById("case-data").textContent);
  const storageKey = `easy-validation-independent-v1:${reviewer}`;
  const blank = c => ({case_id:c.case_id, reviewer, review_status:"pending", verdict:null,
    defect_type:null, one_glance:null, no_clipping_or_occlusion:null, notes:"", reviewed_at:null});
  let results = cases.map(blank);
  let storageWarning = "";
  const status = document.getElementById("progress");
  const allowed = {normal:["none"], defect:["direction","axis","offset","other"], uncertain:["uncertain"]};
  const complete = r => allowed[r.verdict]?.includes(r.defect_type) &&
    typeof r.one_glance === "boolean" && typeof r.no_clipping_or_occlusion === "boolean";
  const validBackup = rows => {
    const keys = Object.keys(blank(cases[0])).sort().join();
    if (rows.length !== cases.length || new Set(rows.map(r=>r.case_id)).size !== cases.length)
      throw Error("备份必须恰好覆盖全部24例，不可包含重复ID。");
    const known = new Set(cases.map(c=>c.case_id));
    for (const r of rows) {
      if (!known.has(r.case_id) || r.reviewer !== reviewer || Object.keys(r).sort().join() !== keys)
        throw Error("只能导入本人、同一组案例的结果。");
      if (r.verdict !== null && !Object.hasOwn(allowed,r.verdict)) throw Error("判断值无效。");
      if (r.defect_type !== null && !["none","direction","axis","offset","other","uncertain"].includes(r.defect_type)) throw Error("问题类型无效。");
      if (![null,true,false].includes(r.one_glance) || ![null,true,false].includes(r.no_clipping_or_occlusion) || typeof r.notes !== "string") throw Error("质量项或备注无效。");
      if (!["pending","completed"].includes(r.review_status) || (r.review_status === "completed" && (!complete(r) || !r.reviewed_at || Number.isNaN(Date.parse(r.reviewed_at))))) throw Error("完成状态或时间无效。");
    }
    return cases.map(c=>rows.find(r=>r.case_id === c.case_id));
  };
  try { const saved = localStorage.getItem(storageKey); if (saved) results = validBackup(JSON.parse(saved)); }
  catch { storageWarning = "本地存储不可用或备份不匹配，请及时下载。"; }
  const persist = () => {
    try { localStorage.setItem(storageKey, JSON.stringify(results)); }
    catch { storageWarning = "本地存储不可用，请及时下载。"; }
    status.textContent = `${results.filter(r=>r.review_status === "completed").length} / 24 已填写。${storageWarning}`;
  };
  const element = (tag, text, cls) => { const el=document.createElement(tag); if(text)el.textContent=text; if(cls)el.className=cls;return el; };
  const render = () => {
    const main=document.getElementById("cases");main.replaceChildren();
    cases.forEach((c,index)=>{
      const r=results[index], card=element("section",null,"case");
      card.append(element("h2",`${index+1}. ${c.case_id}`));
      const board=element("img",null,"board");board.src=c.board;board.alt=`${c.case_id} Reference / Candidate 对照`;card.append(board);
      const details=element("details");details.append(element("summary","查看输入图（上排 Reference，下排 Candidate；左 rest，右 limit）"));
      const preview=element("img",null,"preview");preview.src=c.input_preview;preview.alt="四张输入状态图";details.append(preview);card.append(details);
      const badge=element("p",null,"status");
      const update=()=>{r.review_status=complete(r)?"completed":"pending";r.reviewed_at=r.review_status==="completed"?new Date().toISOString():null;badge.textContent=r.review_status==="completed"?"已填写（尚需单独交回）":"待填写";badge.classList.toggle("completed",r.review_status==="completed");persist();};
      const radios=(field,title,choices)=>{
        const group=element("fieldset");group.append(element("legend",title));
        choices.forEach(([value,label])=>{const wrapper=element("label"),input=element("input");input.type="radio";input.name=`${c.case_id}:${field}`;input.checked=r[field]===value;input.addEventListener("change",()=>{r[field]=value;if(field==="verdict"){r.defect_type=value==="normal"?"none":value==="uncertain"?"uncertain":null;update();render();}else update();});wrapper.append(input,document.createTextNode(` ${label}`));group.append(wrapper);});
        card.append(group);
      };
      radios("verdict","运动是否正常？",[["normal","正常"],["defect","存在问题"],["uncertain","无法判断"]]);
      if(r.verdict==="defect")radios("defect_type","问题类型",[["direction","运动方向"],["axis","转轴或轴向"],["offset","部件偏移"],["other","其他问题"]]);
      radios("one_glance","问题或正常状态是否一眼可判断？",[[true,"是"],[false,"否"]]);
      radios("no_clipping_or_occlusion","是否无明显裁切、遮挡等影响判断的问题？",[[true,"是"],[false,"否"]]);
      const label=element("label","判断依据／无法判断的原因（请独立填写）"),notes=element("textarea");notes.value=r.notes;notes.addEventListener("input",()=>{r.notes=notes.value;update();});label.append(notes);card.append(label,badge);
      badge.textContent=r.review_status==="completed"?"已填写（尚需单独交回）":"待填写";badge.classList.toggle("completed",r.review_status==="completed");main.append(card);
    });persist();
  };
  document.getElementById("download").addEventListener("click",()=>{
    const blob=new Blob([results.map(r=>JSON.stringify(r)).join("\n")+"\n"],{type:"application/x-ndjson;charset=utf-8"});
    const url=URL.createObjectURL(blob),link=element("a");link.href=url;link.download=`${reviewer}.validation-return.jsonl`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.getElementById("import").addEventListener("change",async event=>{
    const file=event.target.files[0];if(!file)return;
    try {const rows=(await file.text()).split(/\r?\n/).filter(s=>s.trim()).map(s=>JSON.parse(s));results=validBackup(rows);render();}
    catch(error){alert(error.message);}event.target.value="";
  });render();
})();
