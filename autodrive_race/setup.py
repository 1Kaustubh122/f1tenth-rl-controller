import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "autodrive_race"
BASE_DIR = Path(__file__).resolve().parent


def collect_glob(pattern: str):
    return [str(path) for path in BASE_DIR.glob(pattern)]


def collect_tree(root: Path, install_prefix: str):
    data_files = []
    if not root.exists():
        return data_files

    for directory, _, filenames in os.walk(root):
        if not filenames:
            continue
        destination = Path(install_prefix) / Path(directory).relative_to(root)
        sources = [str(Path(directory) / filename) for filename in filenames]
        data_files.append((str(destination), sources))
    return data_files


data_files = [
    ("share/ament_index/resource_index/packages", [str(BASE_DIR / "resource" / package_name)]),
    (os.path.join("share", package_name), [str(BASE_DIR / "package.xml")]),
    (os.path.join("share", package_name, "launch"), collect_glob("launch/*.launch.py")),
    (os.path.join("share", package_name, "config"), collect_glob("config/*.yaml")),
    (os.path.join("share", package_name, "autodrive_race"), collect_glob("autodrive_race/*.csv")),
    (os.path.join("share", package_name, "autodrive_race"), collect_glob("autodrive_race/*.py")),
    (os.path.join("share", package_name, "checkpoints"), collect_glob("checkpoints/*")),
    (os.path.join("share", package_name, "best_model"), collect_glob("best_model/*")),
    (os.path.join("share", package_name, "logs"), collect_glob("logs/*")),
    (os.path.join("share", package_name, "maps"), collect_glob("maps/*")),
]
data_files.extend(
    collect_tree(BASE_DIR.parent / "triton_model_repo", os.path.join("share", package_name, "triton_model_repo"))
)

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ppo_training=autodrive_race.ppo_training:main",
            "eval=autodrive_race.eval:main",
            "final_test_training=autodrive_race.final_test_training:main",
            "export_onnx=autodrive_race.export_onnx:main",
            "benchmark_inference=autodrive_race.benchmark_inference:main",
            "build_tensorrt=autodrive_race.build_tensorrt:main",
            "triton_client=autodrive_race.triton_client:main",
        ],
    },
)
