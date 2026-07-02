# 부하테스트 전용 이미지 — 메인 파이프라인 이미지(ghcr.io/seongwonm/ard-milvus, 루트 Dockerfile)와
# 완전히 분리된 별도 이미지다. 루트 Dockerfile/requirements.txt를 건드리지 않으므로
# .github/workflows/build-push.yml의 트리거(paths: Dockerfile, requirements.txt, *.py, bench/**, data/**)를
# 건드리지 않는다 — 즉 이 폴더를 아무리 고쳐도 기존 성능측정용 이미지는 재빌드되지 않는다.
#
# loadtest/** 변경을 push하면 .github/workflows/build-push-loadtest.yml이 자동으로
# ghcr.io/seongwonm/ard-milvus-loadtest:<sha>를 빌드/푸시하고 loadtest/k8s/*.yaml의
# 이미지 태그도 자동으로 갱신한다 — 보통은 수동 빌드가 필요 없다.
#
# 수동 빌드가 필요할 때(로컬 테스트 등, 레포 루트에서 실행·context는 반드시 루트):
#   docker build -f loadtest/Dockerfile -t <registry>/ard-milvus-loadtest:manual .
#   docker push <registry>/ard-milvus-loadtest:manual
FROM python:3.12-slim

# entrypoint.sh가 자기 자신의 milvus/minio를 apply/delete하는 데 필요 (loadtest/k8s/rbac.yaml로 권한 부여)
ARG KUBECTL_VERSION=v1.31.0
RUN apt-get update && apt-get install -y --no-install-recommends curl \
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
