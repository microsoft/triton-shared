from triton.backends.compiler import BaseBackend
from triton._C.libtriton import ir, passes, llvm, triton_shared
from dataclasses import dataclass
from typing import Any
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path

def printc(obj, color="cyan"): #makes things easier to see, will remove later
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_triton_SME_path() -> str:
    return os.getenv("TRITON_SME_PATH", "")


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)




def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call([triton_shared_opt_path, src_path,  
                               "--triton-to-structured", 
                               "--canonicalize", 
                               "--triton-arith-to-linalg", 
                               "--cse", 
                               "--structured-to-memref", 
                               "-o", 
                               dst_path])
        return Path(dst_path).read_text()



def _optimize_ttsharedir(ttsharedir: str):

    if  _get_triton_SME_path() == "":
        return ttsharedir
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "ttshared.mlir")
        sme_first_pass = os.path.join(tmpdir, "sme_first_pass.mlir")
        sme_second_pass = os.path.join(tmpdir, "sme_second_pass.mlir")
        Path(src_path).write_text(ttsharedir)
        triton_sme_opt_path = _get_triton_SME_path()
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        subprocess.check_call([triton_sme_opt_path, src_path, "-sme-conversion" , "-o", sme_first_pass])
        
        
        subprocess.check_call([mlir_opt_path, sme_first_pass,
        "--canonicalize", 
        "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--eliminate-empty-tensors",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--empty-tensor-to-alloc-tensor",
            "--expand-strided-metadata",
            "--arm-sme-vector-legalization",
            "--convert-vector-to-arm-sme",
            "--arm-sme-outer-product-fusion",
            "--arm-sve-legalize-vector-storage",
            "--convert-arith-to-arm-sme",
            "--allocate-arm-sme-tiles",
            "--convert-arm-sme-to-scf",
            "--convert-vector-to-scf=full-unroll",
            "--convert-arith-to-arm-sme",
            "--convert-arith-to-llvm",
            "-o",
            sme_second_pass])
    
        return Path(sme_second_pass).read_text()


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        # TritonShared-MLIR to LLVM-MLIR
        if  _get_triton_SME_path() == "":
            subprocess.check_call([mlir_opt_path, ttshared_path,
            "--convert-linalg-to-affine-loops",
            "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine",
            # "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts",
            # "--debug",
            "-o",
            llmlir_path])

            # "--convert-vector-to-llvm=enable-arm-sve",
            # "--convert-arm-sme-to-llvm",
        else:
            subprocess.check_call([mlir_opt_path, ttshared_path,
            "--canonicalize",
            "--eliminate-empty-tensors",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm", 
             
            "--convert-vector-to-llvm=enable-arm-sve",
            "--convert-arm-sme-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--lower-affine",
            "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts",
            # "--debug",
            "-o", 
            llmlir_path])
        
        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir", 
            "-o",
            llir_path])
        
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        llc_path = _get_llvm_bin_path("llc")
        if  _get_triton_SME_path() == "":
            subprocess.check_call([llc_path, src_path, "-o", dst_path])
        else:
            subprocess.check_call(["/usr/bin/qemu-aarch64-static", llc_path, src_path, "--mattr=+sve,+sme","-o", dst_path])

        return Path(dst_path).read_text()



@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    @staticmethod
    def supports_target(target: tuple):
        return target[0] == 'cpu'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        assert isinstance(target, tuple) and len(target) == 2
        assert isinstance(target[1], str)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target[1]}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # assert False
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(_ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_llir(src))
        stages["cpuasm"] = lambda src, metadata: _llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return f'{1}-{self.target}'
