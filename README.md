# QuadSwarm

Один прогон на 100 агентах от нуля до 1 000 005 120 шагов. На 100 млн шагов обучение сохранили и продолжили с того же чекпоинта.

## Папки

- [`runs/quads100-100m-w16/`](runs/quads100-100m-w16/) — чекпоинты, конфиг и лог до 100 млн шагов.
- [`runs/quads100-1b-w16/`](runs/quads100-1b-w16/) — продолжение того же прогона до 1 млрд шагов. Промежуточные веса лежат в `checkpoint_p0/milestones/`.
- [`evaluation/`](evaluation/) — скрипты тестов и [список проверяемых чекпоинтов](evaluation/manifest.txt).
- [`results/`](results/) — результаты по эпизодам и сводные таблицы.

## Как запустить

Из корня репозитория, с Python 3.11.10 или новее и Git LFS.

```sh
git lfs pull --include="runs/quads100-100m-w16/checkpoint_p0/**,runs/quads100-1b-w16/checkpoint_p0/**"
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git ../quad-swarm-rl
git -C ../quad-swarm-rl checkout e18d6925f73e3d3b35a8acb056817203ff7b8b07
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e ../quad-swarm-rl sample-factory==2.1.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**Тесты**

```sh
python evaluation/evaluate.py \
    --config runs/quads100-1b-w16/config.json \
    --manifest evaluation/manifest.txt --output episodes.csv \
    --agents 24 --episodes 100 --workers 16
python evaluation/summarize.py episodes.csv evaluation/manifest.txt summary.csv
```

**Обучение с нуля на GPU**

```sh
mkdir -p local-runs/quad-swarm
cp -n runs/quads100-1b-w16/config.json local-runs/quad-swarm/config.json
python -m swarm_rl.train --env=quadrotor_multi --experiment=quad-swarm \
    --train_dir=local-runs --restart_behavior=resume --train_for_env_steps=1000000000
```

## Результаты по чекпоинтам

Предварительные данные по 45 одинаковым сценам с 24 агентами. Прочерк означает, что результаты ещё не готовы. [Сводка по трём режимам](results/summary-interim-45-scenes.csv) · [Данные по эпизодам](results/episodes-interim-45-scenes.csv)

| Чекпоинт | Шаги | Успех агентов без отказа | Успех роя без отказа | Успех роя при отказе на 2 с |
|---|---:|---:|---:|---:|
| [`checkpoint_000055200_70656000.pth`](runs/quads100-100m-w16/checkpoint_p0/milestones/checkpoint_000055200_70656000.pth) | 70 656 000 | — | — | — |
| [`best_000076337_97711360_reward_-7.019.pth`](runs/quads100-100m-w16/checkpoint_p0/best_000076337_97711360_reward_-7.019.pth) | 97 711 360 | — | — | — |
| [`checkpoint_000077904_99717120.pth`](runs/quads100-100m-w16/checkpoint_p0/checkpoint_000077904_99717120.pth) | 99 717 120 | — | — | — |
| [`checkpoint_000078128_100003840.pth`](runs/quads100-100m-w16/checkpoint_p0/checkpoint_000078128_100003840.pth) | 100 003 840 | — | — | — |
| [`checkpoint_000078129_100005120.pth`](runs/quads100-100m-w16/checkpoint_p0/checkpoint_000078129_100005120.pth) | 100 005 120 | 5,9% | 0,0% | 0,0% |
| [`checkpoint_000132032_169000960.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000132032_169000960.pth) | 169 000 960 | 57,9% | 0,0% | 0,0% |
| [`checkpoint_000186569_238808320.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000186569_238808320.pth) | 238 808 320 | 55,2% | 0,0% | 0,0% |
| [`checkpoint_000240869_308312320.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000240869_308312320.pth) | 308 312 320 | 16,5% | 0,0% | 0,0% |
| [`checkpoint_000295106_377735680.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000295106_377735680.pth) | 377 735 680 | 79,0% | 2,2% | 2,2% |
| [`checkpoint_000349389_447217920.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000349389_447217920.pth) | 447 217 920 | 87,0% | 4,4% | 6,7% |
| [`checkpoint_000403729_516773120.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000403729_516773120.pth) | 516 773 120 | 93,9% | 24,4% | 20,0% |
| [`best_000437948_560573440_reward_-2.766.pth`](runs/quads100-1b-w16/checkpoint_p0/best_000437948_560573440_reward_-2.766.pth) | 560 573 440 | 87,1% | 0,0% | 2,2% |
| [`checkpoint_000458157_586440960.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000458157_586440960.pth) | 586 440 960 | 91,3% | 17,8% | 17,8% |
| [`checkpoint_000512621_656154880.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000512621_656154880.pth) | 656 154 880 | 93,2% | 22,2% | 20,0% |
| [`checkpoint_000566983_725738240.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000566983_725738240.pth) | 725 738 240 | 95,8% | 51,1% | 35,6% |
| [`checkpoint_000621349_795326720.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000621349_795326720.pth) | 795 326 720 | 95,1% | 40,0% | 33,3% |
| [`checkpoint_000675809_865035520.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000675809_865035520.pth) | 865 035 520 | 92,9% | 22,2% | 15,6% |
| [`checkpoint_000730235_934700800.pth`](runs/quads100-1b-w16/checkpoint_p0/milestones/checkpoint_000730235_934700800.pth) | 934 700 800 | 94,8% | 40,0% | 35,6% |
| [`checkpoint_000780080_998502400.pth`](runs/quads100-1b-w16/checkpoint_p0/checkpoint_000780080_998502400.pth) | 998 502 400 | 91,9% | 15,6% | 11,1% |
| [`checkpoint_000781253_1000003840.pth`](runs/quads100-1b-w16/checkpoint_p0/checkpoint_000781253_1000003840.pth) | 1 000 003 840 | 92,1% | 22,2% | 26,7% |
| [`checkpoint_000781254_1000005120.pth`](runs/quads100-1b-w16/checkpoint_p0/checkpoint_000781254_1000005120.pth) | 1 000 005 120 | 92,4% | 28,9% | 22,2% |
