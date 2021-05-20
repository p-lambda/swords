AVAILABLE_CPUS=$(python3 -c "import os; cpus=os.sched_getaffinity(0); print(','.join(map(str,cpus)))")
AVAILABLE_GPUS=$(nvidia-smi -L | python3 -c "import sys; print(','.join([l.strip().split()[-1][:-1] for l in list(sys.stdin)]))")

docker run \
  -it \
  --rm \
  -d \
  --cpuset-cpus ${AVAILABLE_CPUS} \
  --gpus device=${AVAILABLE_GPUS} \
  --name swords \
  -v $(pwd)/../swords:/swords/swords \
  -v $(pwd)/../assets:/swords/assets \
  -v $(pwd)/../notebooks:/home/swords/notebooks \
  -p 8080:8080 \
  -p 8888:8888 \
  chrisdonahue/swords \
  bash
