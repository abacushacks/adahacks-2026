import os

# Avoid Intel OpenMP shared-memory initialization failures on macOS/dev setups.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
