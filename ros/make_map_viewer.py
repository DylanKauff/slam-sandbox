#!/usr/bin/env python3
"""Turn a saved map (.ply) into ONE self-contained HTML viewer.

    python3 make_map_viewer.py maps/room_20260801_155411.ply maps/room.html "Room scan"

The output is a single .html file with the point data embedded (base64) and a
zero-dependency WebGL point-cloud renderer inside it -- no Three.js, no CDN, no
fonts to fetch. That means it works dropped straight into a GitHub Pages site
(inline via <iframe>, or just linked), opened from disk, or shared as a link.

Handles binary or ASCII PLY, with or without per-vertex colour; if the cloud has
no colour it's shaded by height. Points are centered and scaled to a unit box so
the camera framing is automatic for any map.
"""
import base64
import struct
import sys


def read_ply(path):
    """Return (xyz float32 bytes, rgb uint8 bytes or None, n). Minimal PLY: the
    maps we export are xyz [+ rgb] vertex lists, binary_little_endian or ascii."""
    data = open(path, "rb").read()
    he = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:he].decode("ascii", "ignore")
    n = 0
    props = []            # (name, type)
    fmt_ascii = "ascii" in header
    for line in header.splitlines():
        t = line.split()
        if len(t) >= 3 and t[0] == "element" and t[1] == "vertex":
            n = int(t[2])
        elif len(t) >= 3 and t[0] == "property":
            props.append((t[-1], t[-2]))
    names = [p[0] for p in props]
    xi = [names.index(a) for a in ("x", "y", "z")]
    has_rgb = all(c in names for c in ("red", "green", "blue"))
    ci = [names.index(c) for c in ("red", "green", "blue")] if has_rgb else None

    import numpy as np
    if fmt_ascii:
        vals = np.array(data[he:].split(), dtype=np.float64).reshape(n, len(props))
    else:
        # assume all properties are 4-byte (float x/y/z) except uchar colours
        sizes = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1,
                 "int": 4, "short": 2, "ushort": 2}
        typ = {"float": "f", "float32": "f", "double": "d", "uchar": "B",
               "uint8": "B", "int": "i", "short": "h", "ushort": "H"}
        rec = "<" + "".join(typ[p[1]] for p in props)
        stride = struct.calcsize(rec)
        buf = data[he:he + n * stride]
        arr = np.frombuffer(buf, dtype=np.dtype(
            [(f"c{k}", "<" + typ[p[1]]) for k, p in enumerate(props)]), count=n)
        vals = np.stack([arr[f"c{k}"].astype(np.float64) for k in range(len(props))], 1)

    xyz = vals[:, xi].astype(np.float32)
    rgb = vals[:, ci].astype(np.uint8) if ci else None
    return xyz, rgb, n


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#0a0e14; --bg2:#0f141c; --panel:rgba(18,24,34,.72); --line:rgba(120,170,210,.16);
    --ink:#e6edf5; --muted:#8aa0b6; --accent:#38e1c8; --accent2:#5aa9ff;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:
    radial-gradient(120% 120% at 70% 0%, var(--bg2), var(--bg) 60%);
    color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    overflow:hidden}
  #gl{position:fixed;inset:0;display:block;width:100%;height:100%;cursor:grab;touch-action:none}
  #gl:active{cursor:grabbing}
  .mono{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace}
  .hud{position:fixed;top:16px;left:16px;padding:14px 16px;border-radius:12px;
    background:var(--panel);border:1px solid var(--line);backdrop-filter:blur(10px);
    box-shadow:0 8px 30px rgba(0,0,0,.35);max-width:min(78vw,320px)}
  .hud h1{margin:0;font-size:15px;font-weight:600;letter-spacing:.2px;text-wrap:balance}
  .hud .sub{margin-top:3px;font-size:11px;color:var(--muted);letter-spacing:.4px;
    text-transform:uppercase}
  .stats{margin-top:10px;display:flex;gap:16px;font-size:12px;color:var(--muted)}
  .stats b{display:block;color:var(--ink);font-size:15px;font-weight:600;
    font-variant-numeric:tabular-nums}
  .controls{position:fixed;bottom:16px;left:16px;right:16px;display:flex;gap:8px;
    align-items:center;flex-wrap:wrap;pointer-events:none}
  .controls>*{pointer-events:auto}
  .btn{font:inherit;font-size:12px;color:var(--ink);background:var(--panel);
    border:1px solid var(--line);border-radius:9px;padding:8px 12px;cursor:pointer;
    backdrop-filter:blur(10px);transition:border-color .15s,color .15s}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .btn[aria-pressed=true]{border-color:var(--accent);color:var(--accent)}
  .slider{display:flex;align-items:center;gap:8px;background:var(--panel);
    border:1px solid var(--line);border-radius:9px;padding:6px 12px;backdrop-filter:blur(10px);
    font-size:11px;color:var(--muted)}
  .slider input{accent-color:var(--accent);width:96px}
  .hint{margin-left:auto;font-size:11px;color:var(--muted)}
  @media (max-width:560px){.hint{display:none}}
</style></head>
<body>
<canvas id="gl"></canvas>
<div class="hud">
  <h1>__TITLE__</h1>
  <div class="sub">on-device SLAM · point cloud</div>
  <div class="stats mono">
    <div><b id="npts">—</b>points</div>
    <div><b id="ext">—</b>extent</div>
  </div>
</div>
<div class="controls">
  <button class="btn" id="spin" aria-pressed="true">Auto-spin</button>
  <button class="btn" id="axis">Color: height</button>
  <button class="btn" id="reset">Reset view</button>
  <label class="slider mono">size <input id="size" type="range" min="1" max="6" step="0.2" value="2.4"></label>
  <span class="hint mono">drag to orbit · scroll to zoom</span>
</div>
<script>
const DATA="__DATA__", HASCOL=__HASCOL__, COLDATA="__COLDATA__";
const BOUNDS=__BOUNDS__, NPTS=__NPTS__, EXTENT="__EXTENT__";
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u8(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const pts=b64f32(DATA), cols=HASCOL?b64u8(COLDATA):null;

const cv=document.getElementById("gl");
const gl=cv.getContext("webgl",{antialias:true,alpha:true});
document.getElementById("npts").textContent=NPTS.toLocaleString();
document.getElementById("ext").textContent=EXTENT;

const vs=`attribute vec3 p; attribute vec3 c;
uniform mat4 uMVP; uniform float uSize; uniform vec2 uH; uniform int uAxis; uniform int uMode;
varying float vT; varying vec3 vC;
void main(){ gl_Position=uMVP*vec4(p,1.0);
  float h = uAxis==0?p.x : (uAxis==1?p.y : p.z);
  vT = clamp((h-uH.x)/max(uH.y-uH.x,1e-4),0.0,1.0);
  vC = c;
  gl_PointSize = clamp(uSize*260.0/gl_Position.w, 1.0, 22.0); }`;
const fs=`precision mediump float; varying float vT; varying vec3 vC; uniform int uMode;
vec3 ramp(float t){ // deep-blue -> cyan -> green -> amber -> red
  vec3 a=vec3(0.13,0.20,0.55),b=vec3(0.09,0.72,0.79),c=vec3(0.35,0.85,0.38),
       d=vec3(0.98,0.79,0.22),e=vec3(0.92,0.27,0.24);
  if(t<0.25)return mix(a,b,t/0.25);
  if(t<0.5) return mix(b,c,(t-0.25)/0.25);
  if(t<0.75)return mix(c,d,(t-0.5)/0.25);
  return mix(d,e,(t-0.75)/0.25);}
void main(){ vec2 q=gl_PointCoord*2.0-1.0; float r=dot(q,q); if(r>1.0)discard;
  vec3 col = uMode==1 ? vC : ramp(vT);
  float shade = 1.0-0.35*r;
  gl_FragColor=vec4(col*shade,1.0); }`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(o));return o;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(prog);gl.useProgram(prog);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,pts,gl.STATIC_DRAW);
const aP=gl.getAttribLocation(prog,"p");gl.enableVertexAttribArray(aP);gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
const aC=gl.getAttribLocation(prog,"c");
if(cols){const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);
  const cf=new Float32Array(cols.length);for(let i=0;i<cols.length;i++)cf[i]=cols[i]/255;
  gl.bufferData(gl.ARRAY_BUFFER,cf,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(aC);gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);}
else{gl.disableVertexAttribArray(aC);gl.vertexAttrib3f(aC,1,1,1);}
const uMVP=gl.getUniformLocation(prog,"uMVP"),uSize=gl.getUniformLocation(prog,"uSize"),
  uH=gl.getUniformLocation(prog,"uH"),uAxis=gl.getUniformLocation(prog,"uAxis"),uMode=gl.getUniformLocation(prog,"uMode");
gl.enable(gl.DEPTH_TEST);gl.clearColor(0,0,0,0);

// --- tiny mat4 ---
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return[t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];}
function mul(A,B){const C=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=A[k*4+j]*B[i*4+k];C[i*4+j]=s;}return C;}
function look(e,c,u){let z=sub(e,c);z=norm(z);let x=norm(cross(u,z)),y=cross(z,x);
  return[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1];}
function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];}

// --- camera / interaction ---
let yaw=0.6,pitch=0.5,dist=2.6,spin=true,axis=2,mode=HASCOL?1:0,psize=2.4;
const D={yaw,pitch,dist};
function resize(){const d=window.devicePixelRatio||1;cv.width=cv.clientWidth*d;cv.height=cv.clientHeight*d;gl.viewport(0,0,cv.width,cv.height);}
window.addEventListener("resize",resize);resize();
let drag=false,px=0,py=0;
cv.addEventListener("pointerdown",e=>{drag=true;spin=false;setSpin();px=e.clientX;py=e.clientY;cv.setPointerCapture(e.pointerId);});
cv.addEventListener("pointerup",()=>drag=false);
cv.addEventListener("pointermove",e=>{if(!drag)return;yaw+=(e.clientX-px)*0.008;pitch+=(e.clientY-py)*0.008;
  pitch=Math.max(-1.5,Math.min(1.5,pitch));px=e.clientX;py=e.clientY;});
cv.addEventListener("wheel",e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0011);dist=Math.max(0.4,Math.min(9,dist));},{passive:false});

const $=id=>document.getElementById(id);
function setSpin(){$("spin").setAttribute("aria-pressed",spin);}
$("spin").onclick=()=>{spin=!spin;setSpin();};
$("reset").onclick=()=>{yaw=0.6;pitch=0.5;dist=2.6;spin=true;setSpin();};
$("size").oninput=e=>psize=parseFloat(e.target.value);
const AX=["height (Z)","width (X)","depth (Y)"],AXV=[2,0,1];let axidx=0;
$("axis").onclick=()=>{if(HASCOL&&mode===1){mode=0;}else{axidx=(axidx+1)%3;axis=AXV[axidx];if(HASCOL&&axidx===0){mode=1;}}
  $("axis").textContent = (HASCOL&&mode===1)?"Color: scan":("Color: "+AX[axidx]);};

const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
function frame(){
  if(spin&&!reduce)yaw+=0.0032;
  const a=cv.width/cv.height;
  const cp=Math.cos(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw),sp=Math.sin(pitch);
  const eye=[dist*cp*sy,dist*sp,dist*cp*cy];
  const mvp=mul(persp(1.05,a,0.01,100),look(eye,[0,0,0],[0,1,0]));
  gl.uniformMatrix4fv(uMVP,false,new Float32Array(mvp));
  gl.uniform1f(uSize,psize);gl.uniform2f(uH,BOUNDS[axis][0],BOUNDS[axis][1]);
  gl.uniform1i(uAxis,axis);gl.uniform1i(uMode,mode);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.POINTS,0,NPTS);
  requestAnimationFrame(frame);
}
$("axis").textContent=(HASCOL&&mode===1)?"Color: scan":"Color: height";
setSpin();frame();
</script>
</body></html>"""


def main():
    if len(sys.argv) < 3:
        print('usage: make_map_viewer.py <in.ply> <out.html> [title]'); sys.exit(1)
    import numpy as np
    inp, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "SLAM map"

    xyz, rgb, n = read_ply(inp)
    # center + scale to a unit box (camera framing is then automatic)
    c = (xyz.max(0) + xyz.min(0)) / 2.0
    xyz = xyz - c
    scale = float(np.abs(xyz).max()) or 1.0
    xyz = (xyz / scale).astype(np.float32)
    ext = (xyz.max(0) - xyz.min(0)) * scale       # metres, original
    bounds = [[float(xyz[:, k].min()), float(xyz[:, k].max())] for k in range(3)]

    html = (HTML
            .replace("__TITLE__", title)
            .replace("__DATA__", base64.b64encode(xyz.tobytes()).decode())
            .replace("__HASCOL__", "true" if rgb is not None else "false")
            .replace("__COLDATA__", base64.b64encode(rgb.tobytes()).decode() if rgb is not None else "")
            .replace("__BOUNDS__", str(bounds))
            .replace("__NPTS__", str(n))
            .replace("__EXTENT__", f"{ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f}m"))
    open(out, "w").write(html)
    kb = len(html) / 1024
    print(f"wrote {out} ({kb:.0f} KB, {n:,} points, colour={'yes' if rgb is not None else 'height'})")


if __name__ == "__main__":
    main()
