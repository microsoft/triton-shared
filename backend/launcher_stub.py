import os, subprocess, tempfile
import importlib.util
import sysconfig

from pathlib import Path


def make_launcher(launcher_src, kernel_placeholder_name):
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    def launch(
        gridX, gridY, gridZ, num_warps, num_ctas, clusterDim0, clusterDim1, clusterDim2,
        shared, stream, cu_function, launch_enter_hook, launch_exit_hook, compiled_kernel,
        *args):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile a kernel every time.
        asm_src = compiled_kernel.asm["cpuasm"]
        launcher_src = '''
{launcher_src}
'''.replace("{kernel_placeholder_name}", compiled_kernel.metadata["name"])
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_src_path = os.path.join(tmpdir, "kernel.s")
            launcher_src_path = os.path.join(tmpdir, "main.cxx")
            so_path = os.path.join(tmpdir, "kernel.so")
            Path(asm_src_path).write_text(asm_src)
            Path(launcher_src_path).write_text(launcher_src)
            # Compile it together.
            subprocess.check_call(["g++", launcher_src_path, asm_src_path, f"-I{py_include_dir}", f"-I{Path(__file__).resolve().parent}", "-shared", "-fPIC", "-o", so_path])

            # Load and launch the compiled kernel.
            spec = importlib.util.spec_from_file_location("__triton_shared_ref_cpu_kernel_launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, compiled_kernel, *args)

    return launch
