# Set up a Runpod machine and run the experiments.
#
# A Linux pod is enough and a GPU is not needed: the experiments are CPU-bound (process pools), so pick a pod
# with many vCPUs. If the image lacks them:  apt-get update && apt-get install -y make curl git
#
#   make setup      install uv, Python 3.12 and the dependencies (uv sync)
#   make test       quick checks
#   make start      run every experiment in the background (survives an SSH disconnect); make status / tail / stop
#   make collect    pack logs/ and figures/ into the persistent volume ($(PERSIST))
#   make help       list all targets
#
# Experiments run one after another, each one using every core, so do not use make -j.
# Override on the command line, e.g. make lam EPISODES=300 SEEDS=20

SHELL := /bin/bash -o pipefail
export PATH := $(HOME)/.local/bin:$(PATH)

EPISODES       ?= 150
SEEDS          ?= 10
BANDIT_EPISODES ?= 1000
HYPE_EPISODES  ?= 20
PERSIST        ?= /workspace/tvc-results

RUN   := uv run --no-sync python -m
JAMS  := $(RUN) tvc.fake_jams
TEE    = 2>&1 | tee logs/console/$@.log   # console output kept apart from the result tables in logs/

.PHONY: help setup test all jams bandit pogema lam fine audit scale ablate envs listen hype hype-sceptical \
        hype-compare learners start status tail stop collect clean-logs

help: ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | column -t -s $$'\t'

setup: ## install uv, Python 3.12 and the dependencies
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync
	@mkdir -p logs/console figures

test: ## quick checks of the bandit, the fake-jams testbed and the DQN / PPO learners
	$(RUN) tvc.test_bandit
	$(RUN) tvc.test_fake_jams
	$(RUN) tvc.test_learners

all: test bandit jams learners ## everything, in order

jams: pogema lam fine audit scale ablate envs listen hype hype-sceptical hype-compare ## all fake-jams experiments

bandit: ## multi-agent bandit, naive and calibrated receivers, all credit designs
	@mkdir -p logs/console
	$(RUN) tvc.bandit $(BANDIT_EPISODES) $(TEE)

pogema: ## POGEMA metrics for all credit designs, reputation and shared return (tab:pogema)
	@mkdir -p logs/console
	$(JAMS) pogema $(EPISODES) $(SEEDS) $(TEE)

lam: ## prosocial weight sweep
	@mkdir -p logs/console
	$(JAMS) lam $(EPISODES) $(SEEDS) $(TEE)

fine: ## fine sweep around the predicted threshold
	@mkdir -p logs/console
	$(JAMS) fine $(EPISODES) $(SEEDS) $(TEE)

audit: ## audit-rate sweep: fixed fine vs fine = 2 phi_lo(p)
	@mkdir -p logs/console
	$(JAMS) audit $(EPISODES) $(SEEDS) $(TEE)

scale: ## number of robots and map size
	@mkdir -p logs/console
	$(JAMS) scale $(EPISODES) $(SEEDS) $(TEE)

ablate: ## which parts of TVC are needed
	@mkdir -p logs/console
	$(JAMS) ablate $(EPISODES) $(SEEDS) $(TEE)

envs: ## credit rules, reputation and shared return on the rooms map
	@mkdir -p logs/console
	$(JAMS) envs $(EPISODES) $(SEEDS) rooms $(TEE)

listen: ## how many receivers act on claims (rho sweep)
	@mkdir -p logs/console
	$(JAMS) listen $(EPISODES) $(SEEDS) rooms $(TEE)

hype: ## group deliveries as more robots hype, naive receivers
	@mkdir -p logs/console
	$(JAMS) hype $(HYPE_EPISODES) $(SEEDS) $(TEE)

hype-sceptical: ## the same with sceptical receivers
	@mkdir -p logs/console
	$(JAMS) hype $(HYPE_EPISODES) $(SEEDS) sceptical $(TEE)

hype-compare: hype hype-sceptical ## one figure from the two hype tables
	$(JAMS) hype_compare

learners: ## REINFORCE, DQN and PPO under every credit rule
	@mkdir -p logs/console
	$(RUN) tvc.fake_jams_learners $(EPISODES) $(SEEDS) $(TEE)

start: ## run `make all` in the background
	@mkdir -p logs/console
	nohup $(MAKE) all > logs/console/run_all.log 2>&1 & echo $$! > logs/console/run_all.pid
	@echo "started, pid $$(cat logs/console/run_all.pid); follow it with: make tail"

status: ## what is running and the last log lines
	@pgrep -af "python -m tvc" || echo "no experiment running"
	@tail -n 3 logs/console/run_all.log 2>/dev/null || true

tail: ## follow the background run
	tail -f logs/console/run_all.log

stop: ## stop the background run
	-kill $$(cat logs/console/run_all.pid) 2>/dev/null
	-pkill -f "python -m tvc"

collect: ## pack logs/ and figures/ into $(PERSIST) (a pod's container disk is lost when the pod is deleted)
	mkdir -p $(PERSIST)
	tar czf $(PERSIST)/tvc-results-$$(date +%Y%m%d-%H%M).tgz logs figures
	@ls -lh $(PERSIST) | tail -n 3

clean-logs: ## delete the console logs in logs/console/ (not the result tables)
	rm -rf logs/console
