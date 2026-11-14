#!/usr/bin/env python
# utils/download_wenzhou_data.py
"""Download wenzhou data."""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================

# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "Wenzhou series battery degradation datasets (April_2026)"


ONEDRIVE_URLS = {
    "main": "https://1drv.ms/f/s!AnQLciP1URipksZQPfoVLhdf67Y8mg",
}


TORRENT_FILES = {
    "Wenzhou Sodium-ion Battery Degradation Data.torrent": {
        "description": "钠离子电池老化数据 (核心训练集)",
        "size_estimate": "~350 MB",
        "category": "sodium-ion",
    },
    "Wenzhou Cold Curse Battery Data.torrent": {
        "description": "冷诅咒电池数据 — 141电芯全生命周期 (迁移学习源域)",
        "size_estimate": "~8 GB",
        "category": "cold-curse",
    },
    "Wenzhou Randomized Battery Data.torrent": {
        "description": "随机工况数据 — 多温度/多倍率 (泛化增强)",
        "size_estimate": "~2 GB",
        "category": "randomized",
    },
    "Wenzhou Pack Degradation Data.torrent": {
        "description": "Pack 级老化数据 (Pack 级验证)",
        "size_estimate": "~500 MB",
        "category": "pack",
    },
    "Prognosis Enabled SOC Estimation.torrent": {
        "description": "SOC 估计辅助数据",
        "size_estimate": "~100 MB",
        "category": "soc",
    },
}

# ============================================================

# ============================================================

def check_aria2() -> Optional[str]:
    """Check aria2."""
    aria2 = shutil.which('aria2c') or shutil.which('aria2')
    if aria2:
        logger.info(f"✓ 找到 aria2: {aria2}")
    return aria2


def check_libtorrent() -> bool:
    """Check libtorrent."""
    try:
        import libtorrent
        logger.info(f"✓ libtorrent 已安装: {libtorrent.__version__}")
        return True
    except ImportError:
        logger.info("✗ libtorrent 未安装")
        return False


def download_via_aria2(torrent_path: str, output_dir: str) -> bool:
    """Download via aria2."""
    aria2 = check_aria2()
    if not aria2:
        logger.error("未找到 aria2。请安装: https://aria2.github.io/")
        return False


    tracker_list = _extract_trackers(torrent_path)

    cmd = [
        aria2,
        '--bt-metadata-only=false',
        '--seed-time=0',
        '--max-connection-per-server=16',
        '--split=8',
        '--min-split-size=1M',
        '--dir', output_dir,
    ]

    if tracker_list:
        tracker_args = ','.join(tracker_list)
        cmd += ['--bt-tracker', tracker_args]

    cmd += [torrent_path]

    logger.info(f"  执行: aria2c {' '.join(cmd[3:])}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  aria2 下载失败: {e}")
        return False


def download_via_libtorrent(torrent_path: str, output_dir: str) -> bool:
    """Download via libtorrent."""
    try:
        import libtorrent as lt
        import time as time_mod
    except ImportError:
        logger.error("请先安装 libtorrent: pip install python-libtorrent")
        return False

    ses = lt.session()
    ses.listen_on(6881, 6891)


    settings = ses.get_settings()
    settings['user_agent'] = 'python_client/1.0'
    ses.apply_settings(settings)


    info = lt.torrent_info(torrent_path)
    h = ses.add_torrent({'ti': info, 'save_path': output_dir})

    logger.info(f"  开始下载: {h.name()}")
    logger.info(f"  文件数: {info.num_files()}, 总大小: {info.total_size() / 1e6:.1f} MB")

    start_time = time_mod.time()

    while not h.is_seed():
        s = h.status()

        state_str = ['queued', 'checking', 'downloading metadata',
                     'downloading', 'finished', 'seeding', 'allocating'][s.state]

        progress = s.progress * 100
        dl_rate = s.download_rate / 1024
        peers = s.num_peers
        elapsed = time_mod.time() - start_time

        logger.info(
            f'\r  [{state_str}] {progress:5.1f}% | '
            f'{dl_rate:6.1f} KB/s | Peers: {peers:3d} | '
            f'Elapsed: {elapsed:.0f}s',
        )

        if s.state == lt.torrent_status.seeding:
            break

        time_mod.sleep(1)

    logger.info(f"\n  下载完成!")
    return True


def _extract_trackers(torrent_path: str) -> List[str]:
    """Extract trackers."""
    try:
        with open(torrent_path, 'rb') as f:
            data = f.read().decode('latin-1', errors='ignore')


        import re

        trackers = re.findall(r'(?:udp|https?)://[^\s:"]+:\d+/announce', data)

        trackers = list(set(trackers))
        logger.info(f"  从种子提取 {len(trackers)} 个 tracker")
        return trackers
    except Exception:
        return []


def download_torrent_dataset(dataset_key: str = None,
                             output_dir: str = None) -> bool:
    """Download torrent dataset."""
    if output_dir is None:
        output_dir = str(DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)


    if dataset_key:
        targets = {k: v for k, v in TORRENT_FILES.items()
                   if v['category'] == dataset_key}
    else:
        targets = TORRENT_FILES

    if not targets:
        logger.error(f"未找到 category='{dataset_key}' 的 torrent 文件")
        return False

    logger.info(f"准备下载 {len(targets)} 个数据集...")
    for fname, info in targets.items():
        logger.info(f"  - {fname}: {info['description']} (~{info['size_estimate']})")

    success = True
    for fname, info in targets.items():
        torrent_path = os.path.join(DATA_DIR, fname) if not os.path.isabs(fname) else fname
        if not os.path.exists(torrent_path):
            logger.error(f"  种子文件不存在: {torrent_path}")
            success = False
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"下载: {info['description']}")
        logger.info(f"{'='*60}")


        if check_aria2():
            ok = download_via_aria2(torrent_path, output_dir)
        elif check_libtorrent():
            ok = download_via_libtorrent(torrent_path, output_dir)
        else:
            logger.error(
                "没有可用的下载工具!\n"
                "请安装以下任一:\n"
                "  1. aria2:  https://aria2.github.io/  (推荐)\n"
                "  2. libtorrent (Python):  pip install python-libtorrent\n"
                "\n"
                "或使用手动下载方式:\n"
                "  python utils/download_wenzhou_data.py --method manual"
            )
            return False

        if not ok:
            success = False

    return success


# ============================================================

# ============================================================

def print_manual_instructions():
    """Print manual instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Wenzhou 电池老化数据集 — 手动下载指引                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  方式 A: OneDrive 直链下载 (推荐)                                  ║
║  ─────────────────────────────                                    ║
║  {onedrive_url:<56s} ║
║                                                                    ║
║  方式 B: 种子下载 (需要 BitTorrent 客户端)                         ║
║  ────────────────────────────────────────                          ║
║  种子文件已下载到:                                                  ║
║    {data_dir}                          ║
║                                                                    ║
║  使用任意 BT 客户端 (qBittorrent / Transmission / μTorrent)        ║
║  打开 .torrent 文件即可开始下载。                                   ║
║                                                                    ║
║  方式 C: 网盘下载                                                   ║
║  ────────────────                                                  ║
║  https://wwqn.lanzoul.com/b00mpeez5c                               ║
║  提取码: dr1x                                                      ║
║                                                                    ║
║  ────────────────────────────────────────                          ║
║  数据集来源:                                                        ║
║    GitHub:      https://github.com/lvdongzhen                      ║
║    ResearchGate: https://www.researchgate.net/profile/Dongzhen-Lyu ║
║    License:      CC BY-ND 4.0                                      ║
║                                                                    ║
║  下载完成后将数据放入:                                              ║
║    {data_dir}                          ║
║                                                                    ║
║  推荐下载优先级 (按对我们项目的价值):                                ║
║    1. Wenzhou Sodium-ion Battery Degradation Data (核心!)          ║
║    2. Wenzhou Cold Curse Battery Data (迁移学习源域)                ║
║    3. Wenzhou Randomized Battery Data (泛化增强)                    ║
║    4. Wenzhou Pack Degradation Data (Pack 级验证)                   ║
║    5. Prognosis Enabled SOC Estimation (辅助)                       ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(
        onedrive_url=ONEDRIVE_URLS['main'],
        data_dir=DATA_DIR,
    ))


# ============================================================

# ============================================================

def check_data_integrity() -> Dict[str, any]:
    """Check data integrity."""
    result = {
        'total_files': 0,
        'total_size_mb': 0.0,
        'datasets_found': [],
        'missing': [],
        'can_proceed': False,
    }

    if not DATA_DIR.exists():
        result['missing'] = list(TORRENT_FILES.keys())
        return result


    data_files = []
    for f in DATA_DIR.rglob('*'):
        if f.is_file() and f.suffix.lower() not in ['.torrent']:
            data_files.append(f)

    result['total_files'] = len(data_files)
    result['total_size_mb'] = sum(f.stat().st_size for f in data_files) / 1e6


    found_categories = set()
    for f in data_files:
        fname = f.name.lower()
        if any(kw in fname for kw in ['h-1-2', 'h-2-1', 'h-5-1', 'bs-11', 'sodium']):
            found_categories.add('sodium-ion')
        if any(kw in fname for kw in ['dongzhen', 'cold', 'curse']):
            found_categories.add('cold-curse')
        if any(kw in fname for kw in ['random', 'batch']):
            found_categories.add('randomized')
        if any(kw in fname for kw in ['pack', 'module']):
            found_categories.add('pack')

    result['datasets_found'] = list(found_categories)
    result['missing'] = [c for c in TORRENT_FILES.values()
                         if c['category'] not in found_categories]
    result['can_proceed'] = len(found_categories) > 0

    return result


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wenzhou 电池老化数据集下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python utils/download_wenzhou_data.py --method manual       # Show manual download instructions.
  python utils/download_wenzhou_data.py --method torrent      # Download all selected torrents.
  python utils/download_wenzhou_data.py --method torrent --dataset sodium-ion  # Download sodium-ion data only.
  python utils/download_wenzhou_data.py --check               # Check data integrity.
        """
    )

    parser.add_argument('--method', type=str, default='manual',
                       choices=['manual', 'torrent', 'onedrive'],
                       help='下载方式: manual(显示指引) | torrent(P2P下载) | onedrive(直链)')

    parser.add_argument('--dataset', type=str, default=None,
                       choices=['sodium-ion', 'cold-curse', 'randomized', 'pack', 'soc'],
                       help='仅下载指定数据集 (默认: 全部)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='数据输出目录 (默认: Wenzhou series battery degradation datasets)')

    parser.add_argument('--check', action='store_true',
                       help='仅检查数据完整性，不下载')

    args = parser.parse_args()


    if args.check:
        result = check_data_integrity()
        print(f"\n数据完整性报告:")
        print(f"  文件总数: {result['total_files']}")
        print(f"  总大小: {result['total_size_mb']:.1f} MB")
        print(f"  已发现数据集: {result['datasets_found'] or '无'}")
        if result['missing']:
            print(f"  缺失数据集:")
            for m in result['missing']:
                print(f"    - {m['description']}")
        print(f"  可进行后续处理: {'✓ 是' if result['can_proceed'] else '✗ 否 — 需要钠离子电池数据'}")
        return


    if args.method == 'manual':
        print_manual_instructions()
    elif args.method == 'torrent':
        success = download_torrent_dataset(args.dataset, args.output_dir)
        if success:
            logger.info("\n✓ 下载完成!")
        else:
            logger.error("\n✗ 下载过程中出现错误，请查看日志")
    elif args.method == 'onedrive':
        logger.info(f"请用浏览器打开 OneDrive 链接下载: {ONEDRIVE_URLS['main']}")
        print_manual_instructions()


if __name__ == '__main__':
    main()
