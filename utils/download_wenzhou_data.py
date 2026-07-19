#!/usr/bin/env python
# utils/download_wenzhou_data.py
"""
Wenzhou 电池老化数据集 — 下载与提取工具
==============================================

提供三种方式获取数据:

  方式 1 (推荐): OneDrive 直链下载
    python utils/download_wenzhou_data.py --method onedrive

  方式 2: Torrent 种子下载 (需要安装 aria2 或 libtorrent)
    python utils/download_wenzhou_data.py --method torrent

  方式 3: 手动下载提示
    python utils/download_wenzhou_data.py --method manual

数据下载后将自动解压到指定目录。

来源:
  GitHub:    https://github.com/lvdongzhen
  OneDrive:  https://1drv.ms/f/s!AnQLciP1URipksZQPfoVLhdf67Y8mg
  种子下载:  https://wwqn.lanzoul.com/b00mpeez5c (提取码: dr1x)
"""

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
# 配置
# ============================================================

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据集存放目录
DATA_DIR = BASE_DIR / "Wenzhou series battery degradation datasets (April_2026)"

# OneDrive 分享链接 (可能有多个)
ONEDRIVE_URLS = {
    "main": "https://1drv.ms/f/s!AnQLciP1URipksZQPfoVLhdf67Y8mg",
}

# Torrent 文件信息
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
# 下载器
# ============================================================

def check_aria2() -> Optional[str]:
    """检查系统中是否安装了 aria2"""
    aria2 = shutil.which('aria2c') or shutil.which('aria2')
    if aria2:
        logger.info(f"✓ 找到 aria2: {aria2}")
    return aria2


def check_libtorrent() -> bool:
    """检查 Python libtorrent 是否可用"""
    try:
        import libtorrent
        logger.info(f"✓ libtorrent 已安装: {libtorrent.__version__}")
        return True
    except ImportError:
        logger.info("✗ libtorrent 未安装")
        return False


def download_via_aria2(torrent_path: str, output_dir: str) -> bool:
    """
    使用 aria2 下载 torrent

    aria2 比 libtorrent 更可靠，支持多种 tracker
    """
    aria2 = check_aria2()
    if not aria2:
        logger.error("未找到 aria2。请安装: https://aria2.github.io/")
        return False

    # 获取 tracker 列表
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
    """
    使用 Python libtorrent 下载 torrent (备选方案)
    """
    try:
        import libtorrent as lt
        import time as time_mod
    except ImportError:
        logger.error("请先安装 libtorrent: pip install python-libtorrent")
        return False

    ses = lt.session()
    ses.listen_on(6881, 6891)

    # 设置参数
    settings = ses.get_settings()
    settings['user_agent'] = 'python_client/1.0'
    ses.apply_settings(settings)

    # 加载 torrent
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
    """从 torrent 文件中提取 tracker URL 列表"""
    try:
        with open(torrent_path, 'rb') as f:
            data = f.read().decode('latin-1', errors='ignore')

        # 提取所有 announce URL
        import re
        # 匹配 udp:// 和 http(s):// 开头的 tracker URL
        trackers = re.findall(r'(?:udp|https?)://[^\s:"]+:\d+/announce', data)
        # 去重
        trackers = list(set(trackers))
        logger.info(f"  从种子提取 {len(trackers)} 个 tracker")
        return trackers
    except Exception:
        return []


def download_torrent_dataset(dataset_key: str = None,
                             output_dir: str = None) -> bool:
    """
    下载指定的 torrent 数据集

    Args:
        dataset_key: 数据集标识 ("sodium-ion", "cold-curse", "randomized", "pack", "soc", None=全部)
        output_dir: 输出目录

    Returns:
        bool: 是否全部成功
    """
    if output_dir is None:
        output_dir = str(DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 筛选要下载的 torrent
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

        # 优先使用 aria2
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
# 手动下载指引
# ============================================================

def print_manual_instructions():
    """打印手动下载指引"""
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
# 数据完整性检查
# ============================================================

def check_data_integrity() -> Dict[str, any]:
    """
    检查已下载数据的完整性

    Returns:
        Dict: {
            'total_files': int,
            'total_size_mb': float,
            'datasets_found': List[str],
            'missing': List[str],
            'can_proceed': bool,
        }
    """
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

    # 检查非 torrent 文件
    data_files = []
    for f in DATA_DIR.rglob('*'):
        if f.is_file() and f.suffix.lower() not in ['.torrent']:
            data_files.append(f)

    result['total_files'] = len(data_files)
    result['total_size_mb'] = sum(f.stat().st_size for f in data_files) / 1e6

    # 按数据集分类
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
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wenzhou 电池老化数据集下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python utils/download_wenzhou_data.py --method manual       # 查看手动下载指引
  python utils/download_wenzhou_data.py --method torrent      # 使用 torrent 下载全部
  python utils/download_wenzhou_data.py --method torrent --dataset sodium-ion  # 仅下载钠电数据
  python utils/download_wenzhou_data.py --check               # 检查数据完整性
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

    # 数据完整性检查
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

    # 下载
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
