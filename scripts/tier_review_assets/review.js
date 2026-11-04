"use strict";
(() => {
  const cases = JSON.parse(document.getElementById("case-data").textContent);
  const key = "tier-review-v1:" + cases.map(c => c.case_id).join(":") + ":";

  const decisions = ["easy", "medium", "hard", "clean", "invalid"];
  const reviewer = document.getElementById("reviewer"), progress = document.getElementById("progress");
  const blank = c => ({case_id:c.case_id, reviewer:"", decision:null, notes:"", reviewed_at:null,review_package_sha256:window.REVIEW_PACKAGE_SHA256});
  const el = (tag, text, cls) => {const node=document.createElement(tag);if(text)node.textContent=text;if(cls)node.className=cls;return node;};
  let rows=cases.map(blank), activeReviewer="", warning="", timers=[];
  const persist = () => {
    if(activeReviewer)try {localStorage.setItem(key+activeReviewer,JSON.stringify(rows));}
    catch {warning="本地存储不可用，请及时导出备份。";}
    progress.textContent=`${rows.filter(r=>r.decision!==null).length} / ${cases.length} 已填写。${warning}`;
  };
  const render = () => {
    timers.forEach(clearInterval);timers=[];document.getElementById("cases").replaceChildren();
    cases.forEach((c,i) => {
      const row=rows[i],card=el("section");card.dataset.id=c.case_id;card.append(el("h2",`${i+1}. ${c.case_id}`));
      const motion=el("div",null,"motion"),details=el("details"),board=el("div",null,"board");
      for(const item of c.primary){const fig=el("figure"),img=el("img");img.loading="lazy";img.src=item.path;img.alt=item.caption;fig.append(el("figcaption",item.caption),img);motion.append(fig);}
      details.append(el("summary","展开三态及补充视角"));
      for(const item of c.supplementary){const cell=el("div"),img=el("img");img.loading="lazy";img.src=item.path;img.alt=item.caption;cell.append(el("p",item.caption),img);board.append(cell);}details.append(board);
      const label=el("label","判断 "),select=el("select");
      for(const [value,text] of [["","请选择"],["easy","一眼可见问题"],["medium","仔细比较或思考可确认问题"],["hard","需补充视角或定位细微差异可确认"],["clean","未发现问题／正常"],["invalid","无法判断或显示有问题"]]){const option=el("option",text);option.value=value;select.append(option);}select.value=row.decision||"";select.disabled=!activeReviewer;
      const notes=el("textarea");notes.placeholder="请说明部件、状态及观察依据";notes.value=row.notes;notes.disabled=!activeReviewer;
      const save=()=>{row.decision=select.value||null;row.notes=notes.value;row.reviewer=activeReviewer;row.reviewed_at=row.decision?new Date().toISOString():null;persist();};
      select.onchange=save;notes.oninput=save;label.append(select);card.append(motion,details,label,notes);document.getElementById("cases").append(card);
    });
    persist();
  };
  document.getElementById("start").onclick=()=>{
    const name=reviewer.value.trim();if(!name || name.length>128){alert("请填写本人审核者标识（1–128个字符）。");return;}
    activeReviewer=name;rows=cases.map(blank);warning="";
    try {
      const saved=JSON.parse(localStorage.getItem(key+name)||"null");
      if(saved){const keys=Object.keys(blank(cases[0])).sort().join();
        if(!Array.isArray(saved)||saved.length!==cases.length||!saved.every((r,i)=>r.case_id===cases[i].case_id&&r.reviewer===name&&Object.keys(r).sort().join()===keys&&typeof r.notes==="string"&&(r.decision===null?r.reviewed_at===null:decisions.includes(r.decision)&&typeof r.reviewed_at==="string"&&!Number.isNaN(Date.parse(r.reviewed_at)))))throw Error("invalid saved response");
        rows=saved;
      }
    } catch {warning="本人本地记录不可用，现显示空白；请保留已导出的备份。";}
    rows.forEach(r=>{r.reviewer=name;});render();
  };
  reviewer.oninput=()=>{if(activeReviewer&&reviewer.value.trim()!==activeReviewer){activeReviewer="";rows=cases.map(blank);warning="请点击开始／载入本人审核。";render();}};
  document.getElementById("export").onclick=()=>{
    if(rows.some(r=>r.decision!==null)&&!activeReviewer){alert("请先载入本人审核。");return;}
    const exported=rows.map(r=>({...r,reviewer:activeReviewer}));
    const url=URL.createObjectURL(new Blob([exported.map(r=>JSON.stringify(r)).join("\n")+"\n"],{type:"application/x-ndjson;charset=utf-8"}));
    const a=el("a");a.href=url;a.download="tier-review-return.jsonl";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  render();
})();
