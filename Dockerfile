# Handoff Check in a container, so it runs inside a customer's environment with no network access.
#
#   docker build -t handoff-check .
#   docker run --rm --network none \
#     -v /path/to/clips:/data:ro -v /path/to/out:/out -v /path/to/models:/models:ro \
#     handoff-check check /data --spec /data/spec.yaml --out /out --hand-model /models/hand_landmarker.task --workers 0
#
# --network none proves footage can't leave the machine: the tool never needs the internet.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY handoff ./handoff
RUN pip install --no-cache-dir ".[hands,fleet]" \
    && useradd --create-home handoff

USER handoff
ENTRYPOINT ["handoff"]
CMD ["--help"]
