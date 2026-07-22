"""公开钠电数据下载助手。

小型 Mendeley NFM 文件可自动下载。RWTH 67 电芯整包约 3.44 GB，站点可能
触发浏览器挑战，因此默认只输出服务器下载地址和落盘位置。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

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


def main() -> None:
    parser = argparse.ArgumentParser(description="下载钠电公开数据")
    parser.add_argument("--nfm", action="store_true", help="下载 Mendeley NFM 数据")
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

    if args.show_rwth or not args.nfm:
        print("RWTH 67只商业钠电数据约 3.44 GB，目标目录:")
        print(root / "rwth_67_cells")
        print("数据页面:", RWTH_PAGE)
        print("整包链接:", RWTH_ARCHIVE)
        print("若命令行触发反爬，请浏览器下载后复制到目标目录，再解压。")


if __name__ == "__main__":
    main()
