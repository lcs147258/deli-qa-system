# -*- coding: utf-8 -*-
"""使用本机 Chrome headless 将信息图 HTML 高清渲染为 PNG，并自动裁剪留白。"""
import subprocess
from pathlib import Path
from urllib.request import pathname2url
from PIL import Image, ImageChops

BASE = Path(r"E:\三个项目\05_EduRAG 项目\04-代码\deli_qa_system\ai-qa-classroom")
HTML = BASE / "ai-qa-classroom.html"
RAW = BASE / "_raw_shot.png"
OUT = BASE / "ai-qa-classroom.png"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

url = "file:///" + pathname2url(str(HTML))

cmd = [
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-device-scale-factor=2",
    "--window-size=1040,1900",
    f"--screenshot={RAW}",
    url,
]
subprocess.run(cmd, check=True, capture_output=True)

img = Image.open(RAW).convert("RGB")
bg = Image.new("RGB", img.size, (253, 248, 238))
diff = ImageChops.difference(img, bg)
mask = diff.point(lambda v: 255 if v > 12 else 0)
bbox = mask.getbbox()
pad = 56
left = max(0, bbox[0] - pad)
top = max(0, bbox[1] - pad)
right = min(img.width, bbox[2] + pad)
bottom = min(img.height, bbox[3] + pad)
img.crop((left, top, right, bottom)).save(OUT, "PNG")
print("saved:", OUT, img.crop((left, top, right, bottom)).size)
