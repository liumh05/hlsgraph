"""Self-contained Cytoscape.js + ELK layered HTML adapted from the render skill baseline."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def _vendor(name: str) -> str:
    path = Path(__file__).with_name("vendor") / name
    if not path.is_file():
        raise RuntimeError(f"vendored renderer dependency is missing: {name}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def to_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    cytoscape = _vendor("cytoscape.min.js")
    elk = _vendor("elk.bundled.js")
    return _TEMPLATE.replace("__CYTOSCAPE_B64__", cytoscape).replace(
        "__ELK_B64__", elk).replace("__DATA__", payload)


_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HLSGraph architecture view</title>
<style>
:root{--bg:#f7f8fa;--bar:#171a21;--barfg:#f1f3f5;--panel:#fff;--fg:#17202a;--muted:#687384;
--border:#d8dde5;--compute:#3979b9;--mem:#7b8794;--bottleneck:#d23b35;--edge:#7e8da1;--focus:#111827;--qor:#f0f3f7}
@media(prefers-color-scheme:dark){:root{--bg:#12151a;--bar:#0b0d11;--barfg:#edf1f5;--panel:#1a1f27;--fg:#e8edf3;
--muted:#99a5b4;--border:#303744;--compute:#4f91d2;--mem:#697786;--bottleneck:#e34b44;--edge:#647286;--focus:#f8fafc;--qor:#11161d}}
:root[data-theme="dark"]{--bg:#12151a;--bar:#0b0d11;--barfg:#edf1f5;--panel:#1a1f27;--fg:#e8edf3;
--muted:#99a5b4;--border:#303744;--compute:#4f91d2;--mem:#697786;--bottleneck:#e34b44;--edge:#647286;--focus:#f8fafc;--qor:#11161d}
:root[data-theme="light"]{--bg:#f7f8fa;--bar:#171a21;--barfg:#f1f3f5;--panel:#fff;--fg:#17202a;--muted:#687384;
--border:#d8dde5;--compute:#3979b9;--mem:#7b8794;--bottleneck:#d23b35;--edge:#7e8da1;--focus:#111827;--qor:#f0f3f7}
*{box-sizing:border-box}html,body,#app{margin:0;width:100%;height:100%;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
#app{display:flex;flex-direction:column}#bar{display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--bar);color:var(--barfg);font-size:12px;flex-wrap:wrap}
#bar b{font-size:14px}#bar input,#bar select,#bar button{font:inherit;color:var(--barfg);background:#292e3a;border:1px solid #566070;border-radius:5px;padding:4px 7px}
#bar button{cursor:pointer}.grow{flex:1}.legend{display:inline-flex;align-items:center;gap:5px}.dot{width:11px;height:11px;border-radius:3px}.line{width:24px;border-top:2px solid #9aa}.line.thick{border-top-width:6px}
#main{display:flex;flex:1;min-height:0}#cy{flex:1;min-width:0;background:var(--bg)}#panel{width:350px;max-width:38vw;overflow:auto;background:var(--panel);border-left:1px solid var(--border);padding:13px 15px;font-size:13px;line-height:1.45}
#panel h2{font-size:16px;margin:0 0 4px;word-break:break-word}.muted{color:var(--muted)}.section{margin-top:13px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.box{background:var(--qor);padding:7px 8px;border-radius:6px;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px}
.warn{border-left:3px solid var(--bottleneck);padding:5px 8px;background:var(--qor);margin-top:8px}.badge{display:inline-block;padding:2px 6px;border:1px solid var(--border);border-radius:999px;margin:2px 3px 2px 0;font-size:10px}.row{border-bottom:1px solid var(--border);padding:5px 0}.predicate{font-family:ui-monospace,Consolas,monospace;font-size:11px}#tip{display:none;position:fixed;z-index:10;pointer-events:none;background:#000d;color:#fff;border-radius:5px;padding:6px 9px;font-size:11px;max-width:360px;white-space:pre-wrap}
@media(max-width:800px){#panel{width:290px;max-width:45vw}.legend.optional{display:none}}
</style>
<script>new Function(atob("__CYTOSCAPE_B64__"))();</script>
<script>new Function(atob("__ELK_B64__"))();</script>
</head><body><div id="app">
<div id="bar"><b>HLSGraph</b><span id="header"></span><input id="search" placeholder="search name"/>
<select id="category"><option value="*">all categories</option><option value="compute">compute</option><option value="mem">memory / IO</option></select>
<select id="stage"><option value="*">all stages</option></select><select id="authority"><option value="*">all authorities</option></select>
<button id="theme" title="toggle light/dark theme">theme</button><button id="reset">reset</button><span class="grow"></span>
<span class="legend"><span class="dot" style="background:#d23b35"></span>bottleneck</span><span class="legend optional"><span class="dot" style="background:#3979b9"></span>compute</span><span class="legend optional"><span class="dot" style="background:#7b8794"></span>memory</span><span class="legend optional"><span class="line"></span><span class="line thick"></span>FIFO depth</span></div>
<div id="main"><div id="cy"></div><aside id="panel"><p class="muted">Click a node to focus its neighborhood and inspect stage-scoped observations and evidence. Left to right is hardware dataflow.</p></aside></div></div><div id="tip"></div>
<script>
const DATA=__DATA__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=x=>String(x==null?'':x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const edgeWidth=d=>Math.max(1.5,Math.min(7,1.2+1.4*Math.log2((Number(d)||1)+1)));
const color=n=>n.is_bottleneck?css('--bottleneck'):(n.category==='compute'?css('--compute'):css('--mem'));
const NODE_W=132,NODE_H=48;
const elements={nodes:DATA.nodes.map(n=>({data:{...n,color:color(n)}})),edges:DATA.edges.map(e=>({data:{...e,width_hint:edgeWidth(e.fifo_depth)}}))};
const cy=cytoscape({container:document.getElementById('cy'),elements,wheelSensitivity:.22,boxSelectionEnabled:false,style:[
{selector:'node',style:{width:NODE_W,height:NODE_H,shape:'round-rectangle',label:e=>{const ii=e.data('metrics').achieved_II;return e.data('label')+(ii!=null?'\nII '+ii:'')},'font-size':11,'text-valign':'center','text-halign':'center','text-wrap':'wrap','text-max-width':NODE_W-12,'line-height':1.15,color:'#fff','background-color':'data(color)','border-width':1.5,'border-color':css('--border')}},
{selector:'node[?is_bottleneck]',style:{'border-width':4,'border-color':'#7d1713'}},{selector:'edge',style:{width:'data(width_hint)','curve-style':'taxi','taxi-direction':'horizontal','taxi-turn':'50%','line-color':css('--edge'),'target-arrow-color':css('--edge'),'target-arrow-shape':'triangle','arrow-scale':1.05,opacity:.86}},
{selector:'edge[type="CONTAINS"]',style:{'line-style':'dashed',width:1.3}},{selector:'edge.showlabel',style:{label:e=>e.data('fifo_depth')==null?e.data('type'):'FIFO '+e.data('fifo_depth'),'font-size':10,color:css('--fg'),'text-background-color':css('--panel'),'text-background-opacity':.92,'text-background-padding':2}},
{selector:'.faded',style:{opacity:.10,'text-opacity':.10}},{selector:'.dim',style:{opacity:.06,'text-opacity':0}},{selector:'.hl',style:{'border-width':4,'border-color':css('--focus'),opacity:1}}
]});
async function layout(){const elk=new ELK();const g={id:'root',layoutOptions:{'elk.algorithm':'layered','elk.direction':'RIGHT','elk.layered.spacing.nodeNodeBetweenLayers':'96','elk.spacing.nodeNode':'42','elk.layered.considerModelOrder.strategy':'NODES_AND_EDGES','elk.edgeRouting':'ORTHOGONAL'},children:cy.nodes().map(n=>({id:n.id(),width:NODE_W,height:NODE_H})),edges:cy.edges().map(e=>({id:e.id(),sources:[e.source().id()],targets:[e.target().id()]}))};try{const out=await elk.layout(g),pos={};out.children.forEach(n=>pos[n.id]={x:n.x+n.width/2,y:n.y+n.height/2});cy.layout({name:'preset',positions:pos,fit:true,padding:42}).run()}catch(err){console.warn(err);cy.layout({name:'breadthfirst',directed:true,fit:true,padding:36}).run()}window.hlsgraphRenderReady=true}
layout();
document.getElementById('header').textContent=`${DATA.meta.top||'?'} · ${DATA.meta.view} · ${DATA.nodes.length} nodes`;
const stages=[...new Set(DATA.nodes.map(n=>n.stage).filter(Boolean))].sort();const stage=document.getElementById('stage');stages.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;stage.appendChild(o)});
const authorities=[...new Set(DATA.nodes.map(n=>n.authority).filter(Boolean))].sort();const authority=document.getElementById('authority');authorities.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;authority.appendChild(o)});
const tip=document.getElementById('tip');function move(ev){const o=ev.originalEvent;if(!o)return;tip.style.left=o.clientX+14+'px';tip.style.top=o.clientY+14+'px'}
cy.on('mouseover','node',ev=>{const d=ev.target.data();tip.textContent=d.name+(d.bottleneck_cause?'\n'+d.bottleneck_cause:'');tip.style.display='block';move(ev)});cy.on('mousemove','node',move);cy.on('mouseout','node',()=>tip.style.display='none');
cy.on('mouseover','edge',ev=>{ev.target.addClass('showlabel');const d=ev.target.data();tip.textContent=(d.fifo_depth==null?d.type:'FIFO depth = '+d.fifo_depth)+(d.elem_type?'\n'+d.elem_type:'');tip.style.display='block';move(ev)});cy.on('mousemove','edge',move);cy.on('mouseout','edge',ev=>{ev.target.removeClass('showlabel');tip.style.display='none'});
const panel=document.getElementById('panel');function panelFor(n){const d=n.data(),m=d.metrics||{};let h=`<h2>${esc(d.name)}${d.is_bottleneck?' 🔴':''}</h2><span class="badge">${esc(d.type)}</span><span class="badge">${esc(d.stage)}</span><span class="badge">${esc(d.authority)}</span>`;if(d.bottleneck_cause)h+=`<div class="warn">${esc(d.bottleneck_cause)}</div>`;for(const p of d.display_conflicts||[])h+=`<div class="warn">Conflicting equal-rank observations: ${esc(p)}; no display value selected.</div>`;const metrics=Object.entries(m).filter(([,v])=>v!=null).map(([k,v])=>`${k.padEnd(14)}${v}`).join('\n');h+=`<div class="section">display metrics</div><div class="box">${esc(metrics||'no metric observation')}</div>`;if((d.directives||[]).length){h+='<div class="section">directives on this scope</div>';for(const x of d.directives){h+=`<div class="row"><div class="predicate">${esc(x.kind)} · ${esc(x.state)}</div><div class="box">${esc(JSON.stringify(x.options||{}))}</div><span class="badge">${esc(x.origin||'unknown origin')}</span><span class="badge">${esc(x.completeness)}</span>`;for(const o of x.observations||[])h+=`<div>${esc(o.predicate)} = ${esc(JSON.stringify(o.value))}</div>`;h+='</div>'}}h+='<div class="section">observations</div>';for(const o of d.observations||[])h+=`<div class="row"><div class="predicate">${esc(o.predicate)} = ${esc(JSON.stringify(o.value))}${o.unit?' '+esc(o.unit):''}</div><span class="badge">${esc(o.stage)}</span><span class="badge">${esc(o.authority)}</span></div>`;h+='<div class="section">evidence anchors</div>';for(const a of d.evidence||[])h+=`<div class="row">${esc(a.artifact_id)}${a.start_line?':'+a.start_line:''}${a.ir_location?' · '+esc(a.ir_location):''}</div>`;if((d.diagnostics||[]).length){h+='<div class="section">diagnostics</div>';for(const x of d.diagnostics)h+=`<div class="warn">${esc(x.code)}: ${esc(x.message)}</div>`}panel.innerHTML=h}
cy.on('tap','node',ev=>{cy.elements().addClass('faded');const n=ev.target,nb=n.closedNeighborhood();nb.removeClass('faded').addClass('hl');nb.connectedEdges().addClass('showlabel');panelFor(n)});cy.on('tap',ev=>{if(ev.target===cy){cy.elements().removeClass('faded hl showlabel');panel.innerHTML='<p class="muted">Click a node to inspect evidence.</p>'}});
function filters(){const q=document.getElementById('search').value.trim().toLowerCase(),cat=document.getElementById('category').value,st=stage.value,auth=authority.value;cy.batch(()=>{cy.nodes().forEach(n=>{const hide=(q&&!String(n.data('name')).toLowerCase().includes(q))||(cat!=='*'&&n.data('category')!==cat)||(st!=='*'&&n.data('stage')!==st)||(auth!=='*'&&n.data('authority')!==auth);n.toggleClass('dim',hide)});cy.edges().forEach(e=>e.toggleClass('dim',e.source().hasClass('dim')||e.target().hasClass('dim')))})}
document.getElementById('search').addEventListener('input',filters);document.getElementById('category').addEventListener('change',filters);stage.addEventListener('change',filters);authority.addEventListener('change',filters);document.getElementById('reset').addEventListener('click',()=>{document.getElementById('search').value='';document.getElementById('category').value='*';stage.value='*';authority.value='*';cy.elements().removeClass('dim faded hl showlabel');cy.fit(undefined,42)});
function restyle(){cy.nodes().forEach(n=>n.data('color',color(n.data())));cy.style().selector('edge').style({'line-color':css('--edge'),'target-arrow-color':css('--edge')}).update()}if(window.matchMedia){const mq=matchMedia('(prefers-color-scheme:dark)');mq.addEventListener?mq.addEventListener('change',restyle):mq.addListener(restyle)}
document.getElementById('theme').addEventListener('click',()=>{const root=document.documentElement,current=root.dataset.theme;root.dataset.theme=current==='dark'?'light':current==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark');restyle()});
</script></body></html>'''
