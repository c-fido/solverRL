"""Build solverrl_core pybind11 extension."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _find_python_include() -> str:
    candidates = [
        Path(sysconfig.get_path("include")),
        Path(sys.base_prefix) / "include",
        Path(sys.prefix) / "include",
    ]
    for path in candidates:
        if (path / "Python.h").is_file():
            return str(path)
    searched = ", ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Python development headers not found (Python.h).\n"
        f"Searched: {searched}\n"
        "Install Python with development headers enabled "
        "(on Windows: python.org installer → Include development files)."
    )


def _optional_library_dirs() -> list[str]:
    libdir = sysconfig.get_config_var("LIBDIR")
    dirs: list[str] = []
    if libdir:
        dirs.append(str(libdir))
    libs = Path(sys.base_prefix) / "libs"
    if libs.is_dir():
        dirs.append(str(libs))
    return dirs


include_dir = _find_python_include()
library_dirs = _optional_library_dirs()

ext_modules = [
    Pybind11Extension(
        "solverrl_core",
        [
            "src/core.cpp",
            "src/vocabulary.cpp",
            "src/foil.cpp",
            "src/prolog_emit.cpp",
            "src/keydoor_ground.cpp",
            "src/rule_learner.cpp",
            "src/exact_eval.cpp",
            "src/expand.cpp",
        ],
        include_dirs=["include", pybind11.get_include(), include_dir],
        library_dirs=library_dirs,
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
