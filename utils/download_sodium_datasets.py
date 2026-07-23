"""公开钠电数据下载助手。

小型 Mendeley NFM 文件可自动下载。RWTH 67 电芯整包约 3.44 GB，站点可能
触发浏览器挑战，因此默认只输出服务器下载地址和落盘位置。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.cookiejar
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from models.soh_ai.config import SODIUM_RAW_DATA_DIR


NFM_SODIUM_FILES = {
    "Dataset 1-1.xlsx": "https://data.mendeley.com/public-files/datasets/4mztcdc4gt/files/44475f69-e73e-48e6-afad-cb91b042fd62/file_downloaded",
    "Dataset 1-2.xlsx": "https://data.mendeley.com/public-files/datasets/4mztcdc4gt/files/553f5932-d024-4d12-8b73-19afa4793ba6/file_downloaded",
}
NFM_LITHIUM_CONTROL = {
    "Dataset 2.xlsx": "https://data.mendeley.com/public-files/datasets/4mztcdc4gt/files/4dcd179d-b220-4bb1-b40c-af5f730dbbc5/file_downloaded",
}

RWTH_PAGE = "https://publications.rwth-aachen.de/record/987579/files/"
RWTH_ARCHIVE = (
    "https://publications.rwth-aachen.de/record/987579/files/"
    "Data_Failure_Mode_and_Degradation_Analysis_of_a_Commercial_"
    "Sodium-Ion_Battery_With_Severe_Gassing_Issue.zip?version=1"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _rwth_request(url: str, start: int | None = None, end: int | None = None):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": "APP_INIT=1",
    }
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return Request(url, headers=headers)


def _rwth_size(url: str) -> int:
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    with opener.open(_rwth_request(url), timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "application/zip" not in content_type:
            raise RuntimeError("RWTH 下载接口没有返回 ZIP，请检查站点挑战是否变化")
        size = int(response.headers["Content-Length"])
    return size


def _download_rwth_part(
    url: str, part_path: Path, start: int, end: int, expected_size: int
) -> int:
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return expected_size

    temp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0
    if downloaded > expected_size:
        temp_path.unlink()
        downloaded = 0
    request_start = start + downloaded
    with urlopen(_rwth_request(url, request_start, end), timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"RWTH 分片未返回 HTTP 206: {response.status}")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {request_start}-{end}/"):
            raise RuntimeError(f"RWTH 分片范围不匹配: {content_range}")
        with temp_path.open("ab") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if temp_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"RWTH 分片大小错误: {temp_path.stat().st_size} != {expected_size}"
        )
    os.replace(temp_path, part_path)
    return expected_size


def download_rwth_archive(
    destination: Path, connections: int = 12, part_size_mb: int = 64
) -> None:
    """多连接、可断点恢复地下载 RWTH 官方 ZIP，并执行完整性校验。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_size = _rwth_size(RWTH_ARCHIVE)
    if destination.exists() and destination.stat().st_size == total_size:
        with zipfile.ZipFile(destination) as archive:
            bad_file = archive.testzip()
        if bad_file is None:
            print(f"RWTH ZIP 已存在且校验通过: {destination}")
            return

    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_size = part_size_mb * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, total_size, part_size)):
        end = min(start + part_size - 1, total_size - 1)
        ranges.append((index, start, end))

    print(
        f"RWTH ZIP: {total_size / 1024 ** 3:.2f} GiB, "
        f"{len(ranges)} 个分片, {connections} 路并发"
    )
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        futures = {
            pool.submit(
                _download_rwth_part,
                RWTH_ARCHIVE,
                parts_dir / f"part-{index:04d}",
                start,
                end,
                end - start + 1,
            ): index
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            print(f"下载进度: {completed / total_size:.1%}", flush=True)

    temp_archive = destination.with_suffix(destination.suffix + ".tmp")
    with temp_archive.open("wb") as output:
        for index, _, _ in ranges:
            with (parts_dir / f"part-{index:04d}").open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if temp_archive.stat().st_size != total_size:
        raise RuntimeError("RWTH 合并文件大小错误")
    os.replace(temp_archive, destination)
    with zipfile.ZipFile(destination) as archive:
        bad_file = archive.testzip()
    if bad_file is not None:
        raise RuntimeError(f"RWTH ZIP CRC 校验失败: {bad_file}")
    shutil.rmtree(parts_dir)
    print(f"RWTH ZIP 下载及校验完成: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载钠电公开数据")
    parser.add_argument("--nfm", action="store_true", help="下载 Mendeley NFM 数据")
    parser.add_argument("--rwth", action="store_true", help="下载并校验 RWTH 官方整包")
    parser.add_argument("--connections", type=int, default=12, help="RWTH 并发下载连接数")
    parser.add_argument(
        "--include-lithium-control", action="store_true",
        help="同时下载 Dataset 2 锂电对照（纯钠电训练不会加载）",
    )
    parser.add_argument("--show-rwth", action="store_true", help="显示 RWTH 下载说明")
    parser.add_argument("--root", default=SODIUM_RAW_DATA_DIR)
    args = parser.parse_args()
    root = Path(args.root)

    if args.nfm:
        target = root / "mendeley_nfm"
        files = dict(NFM_SODIUM_FILES)
        if args.include_lithium_control:
            files.update(NFM_LITHIUM_CONTROL)
        for name, url in files.items():
            path = target / name
            if path.exists() and path.stat().st_size > 1024:
                print(f"已存在: {path}")
                continue
            print(f"下载: {name}")
            _download(url, path)

    if args.rwth:
        target = root / "rwth_67_cells"
        archive = target / "rwth_commercial_sodium_ion_aging.zip"
        download_rwth_archive(archive, connections=max(1, args.connections))

    if args.show_rwth or not (args.nfm or args.rwth):
        print("RWTH 67只商业钠电数据约 3.44 GB，目标目录:")
        print(root / "rwth_67_cells")
        print("数据页面:", RWTH_PAGE)
        print("整包链接:", RWTH_ARCHIVE)
        print("若命令行触发反爬，请浏览器下载后复制到目标目录，再解压。")


if __name__ == "__main__":
    main()
