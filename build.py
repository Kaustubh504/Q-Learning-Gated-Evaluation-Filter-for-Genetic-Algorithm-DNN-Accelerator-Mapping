import os

# MAESTRO source is vendored in ./maestro (self-contained: no external clone
# needed). This builds it in place via SCons and symlinks the resulting
# binary into cost_model/maestro, where GAMMA expects to find it.
maestro_dir = "maestro"
dst_path = "cost_model/maestro"

working_path = os.getcwd()
dst_path = os.path.abspath(dst_path)
maestro_bin = os.path.abspath(os.path.join(maestro_dir, "maestro"))

if not os.path.exists(maestro_bin):
    os.chdir(maestro_dir)
    ret = os.system("scons")
    os.chdir(working_path)
    if ret != 0 or not os.path.exists(maestro_bin):
        raise RuntimeError(
            "Failed to build MAESTRO from ./maestro. Check that a C++ "
            "toolchain, SCons, and the Boost libraries (program_options, "
            "filesystem, system) are installed."
        )

if not os.path.exists(dst_path):
    os.symlink(maestro_bin, dst_path)
