# 부하테스트 전용 이미지 — 메인 파이프라인 이미지(ghcr.io/seongwonm/ard-milvus, 루트 Dockerfile)와
# 완전히 분리된 별도 이미지다. 루트 Dockerfile/requirements.txt를 건드리지 않으므로
# .github/workflows/build-push.yml의 트리거(paths: Dockerfile, requirements.txt, *.py, bench/**, data/**)를
# 건드리지 않는다 — 즉 이 폴더를 아무리 고쳐도 기존 성능측정용 이미지는 재빌드되지 않는다.
#
# loadtest/** 변경을 push하면 .github/workflows/build-push-loadtest.yml이 자동으로
# ghcr.io/seongwonm/ard-milvus-loadtest:<sha>를 빌드/푸시하고 loadtest/k8s/*.yaml의
# 이미지 태그도 자동으로 갱신한다 — 보통은 수동 빌드가 필요 없다.
#
# 주의: 이 이미지는 COPY . ./milvus_migration/로 레포 루트 파일(milvus_store.py 등)도
# 포함해서 빌드되지만, build-push-loadtest.yml의 트리거 경로는 loadtest/**만 본다.
# 즉 루트 *.py만 고치면(예: milvus_store.py 버그 수정) 이 워크플로는 자동으로 안 돈다 —
# gh workflow_dispatch로 수동 실행하거나, loadtest/** 안의 파일도 같이 건드려야 한다
# (2026-07-03 실제로 이 문제로 수동 대응함).
#
# 수동 빌드가 필요할 때(로컬 테스트 등, 레포 루트에서 실행·context는 반드시 루트):
#   docker build -f loadtest/Dockerfile -t <registry>/ard-milvus-loadtest:manual .
#   docker push <registry>/ard-milvus-loadtest:manual
FROM python:3.12-slim

# entrypoint.sh가 자기 자신의 milvus/minio를 apply/delete하는 데 필요 (loadtest/k8s/rbac.yaml로 권한 부여).
# tar는 kubectl cp가 pod 안에서 결과를 스트리밍하는 데 필요(bench용 루트 Dockerfile도 동일한
# 이유로 설치함) — 없으면 fetch-results.sh의 kubectl cp가 빈 결과를 만든다(2026-07-07 실측).
ARG KUBECTL_VERSION=v1.31.0
RUN apt-get update && apt-get install -y --no-install-recommends curl tar \
    && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY loadtest/requirements.txt ./requirements-loadtest.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-loadtest.txt

COPY . ./milvus_migration/
COPY data/ /data/

CMD ["sleep", "infinity"]
