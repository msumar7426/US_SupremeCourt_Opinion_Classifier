FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copied before the source so the expensive pip layer stays cached when only
# application code changes.
COPY requirements.txt .

# Every pinned dependency resolves to a prebuilt manylinux wheel on CPython 3.12,
# so no compiler toolchain is needed.
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expects the application to run as UID 1000.
RUN useradd -m -u 1000 user

# --chown is required. A plain COPY preserves the source file mode, so files
# that are not world readable on the build machine stay unreadable to this
# non-root user and gunicorn fails to import app.py.
COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

EXPOSE 7860

# One worker keeps a single copy of the 140 MB model in memory. Threads let
# requests overlap because PyTorch releases the GIL during the forward pass.
# The default 30s timeout is too tight for a long CPU forward pass.
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
