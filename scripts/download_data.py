"""
数据集下载脚本：从 HuggingFace 镜像获取 NEU-DET（钢材表面缺陷）数据集。

用法：
    python scripts/download_data.py                 # 默认下载到 data/NEU-DET
    python scripts/download_data.py --output data   # 指定目录

说明：GHW 官方发布为 IMAGES + VOC 标注，本脚本只取分类任务需要的图片与划分清单，
共 1797 张 jpg（约 30MB），按官方 train/val/test 划分存放。
"""

import argparse
import os
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://hf-mirror.com/datasets/KeenForgeAI/NEU-DET-corrected/resolve/main"
SPLITS = ["train", "val", "test"]


def fetch(url, dest, retries=3):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            return dest
        except Exception as exc:                      # noqa: BLE001
            print("重试 {}/{} 失败: {} -> {}".format(attempt + 1, retries, url, exc))
    raise RuntimeError("下载失败: " + url)


def read_list(url, dest):
    fetch(url, dest)
    with open(dest, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out_root = os.path.join(args.output, "NEU-DET")
    os.makedirs(out_root, exist_ok=True)

    fetch("{}/classes.txt".format(BASE), os.path.join(out_root, "classes.txt"))

    tasks = []
    for split in SPLITS:
        list_path = os.path.join(out_root, "{}.txt".format(split))
        names = read_list("{}/ImageSets/{}.txt".format(BASE), list_path)
        print("{}: {} 张".format(split, len(names)))
        for name in names:
            url = "{}/images/{}/{}.jpg".format(BASE, split, name)
            dest = os.path.join(out_root, "images", split, "{}.jpg".format(name))
            tasks.append((url, dest))

    print("开始下载 {} 张图片 ...".format(len(tasks)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda t: fetch(*t), tasks), 1):
            if i % 200 == 0:
                print("  已下载 {}/{}".format(i, len(tasks)))

    print("完成，数据位于", out_root)


if __name__ == "__main__":
    main()
