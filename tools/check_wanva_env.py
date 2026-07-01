#!/usr/bin/env python3
"""Preflight checks for the LingBot-VA post-training environment."""
import argparse
import importlib.metadata as md
import re
import sys


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def major_minor(version):
    if version is None:
        return None
    match = re.match(r"^(\d+\.\d+)", version)
    return match.group(1) if match else version


def check_import(name):
    try:
        __import__(name)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def expected_torchvision_major_minor(torch_version):
    torch_mm = major_minor(torch_version)
    mapping = {
        "2.5": "0.20",
        "2.6": "0.21",
        "2.7": "0.22",
        "2.8": "0.23",
        "2.9": "0.24",
    }
    return mapping.get(torch_mm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": pkg_version("torch"),
        "torchvision": pkg_version("torchvision"),
        "torchaudio": pkg_version("torchaudio"),
        "diffusers": pkg_version("diffusers"),
        "transformers": pkg_version("transformers"),
        "lerobot": pkg_version("lerobot"),
        "pyarrow": pkg_version("pyarrow"),
    }
    print("LingBot-VA env preflight:")
    for name, version in versions.items():
        print(f"  {name}: {version or 'not installed'}")

    problems = []
    torch_mm = major_minor(versions["torch"])
    if versions["torchaudio"] is not None and major_minor(versions["torchaudio"]) != torch_mm:
        problems.append(
            f"torchaudio {versions['torchaudio']} does not match torch {versions['torch']}"
        )
    expected_torchvision = expected_torchvision_major_minor(versions["torch"])
    if (
        expected_torchvision is not None
        and versions["torchvision"] is not None
        and major_minor(versions["torchvision"]) != expected_torchvision
    ):
        problems.append(
            f"torchvision {versions['torchvision']} does not match torch {versions['torch']} "
            f"(expected torchvision {expected_torchvision}.x)"
        )

    torchaudio_error = check_import("torchaudio") if versions["torchaudio"] else None
    if torchaudio_error:
        problems.append(f"torchaudio import failed: {torchaudio_error}")

    if args.strict:
        expected = {
            "torch": "2.9",
            "torchvision": "0.24",
            "torchaudio": "2.9",
            "diffusers": "0.36",
            "transformers": "4.55",
        }
        for name, expected_mm in expected.items():
            if major_minor(versions[name]) != expected_mm:
                problems.append(
                    f"{name} is {versions[name] or 'not installed'}, expected {expected_mm}.x"
                )
        if sys.version_info[:2] != (3, 10):
            problems.append(
                f"Python is {versions['python']}, LingBot-VA README expects Python 3.10"
            )

    if problems:
        print("\nEnvironment problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("Environment preflight passed")


if __name__ == "__main__":
    main()
