# Sequential Experimentation Engine

A production-oriented experimentation system for evaluating fraud-threshold changes on streaming transaction data using **Redpanda, PostgreSQL, Python, and sequential statistical testing**.

---

## Problem

Traditional A/B experiments often assume a fixed dataset and a fixed sample size.

Real fraud systems work differently:

* transactions arrive continuously
* fraud outcomes may arrive later
* control and treatment decisions affect real operations
* experiments need safety guardrails
* services can crash or restart
* the same streaming event may be delivered more than once
* experiment state must survive failures

Keeping the entire experiment inside Python memory would therefore be unreliable.

This project builds a streaming experimentation engine that processes transaction and fraud-outcome events continuously while preserving experiment state durably.

---

## Why It Matters

Experimentation in financial systems is not only a statistics problem.

For example, changing a fraud threshold might reduce false declines but simultaneously allow more high-value fraud through the system.

A usable experimentation platform therefore needs to combine:

* statistical evidence
* operational KPIs
* guardrails
* sample-ratio monitoring
* delayed outcomes
* durable state
* duplicate protection
* crash recovery

The project explores how statistical experimentation can be moved from a notebook into a more production-oriented event-driven system.

---

## Architecture

```text
Synthetic Transaction Generator
            |
            v
        Redpanda
   +-------------------+
   | transaction.events|
   | fraud.outcomes    |
   +-------------------+
            |
            v
    Python Stream Consumer
            |
            v
      ExperimentEngine
      /       |       \
     /        |        \
Sequential   SRM      Guardrails
Inference   Checks
     \        |        /
      \       |       /
            |
            v
        PostgreSQL
     Events + Snapshots
            |
            v
      Recovery / Replay
```

### Event-processing flow

```text
Redpanda Event
      |
      v
ExperimentEngine
      |
      v
Persist event + latest state
      |
      v
PostgreSQL COMMIT
      |
      v
Kafka offset COMMIT
```

PostgreSQL is committed **before** the Kafka offset is acknowledged.

If the process crashes after PostgreSQL commits but before the Kafka offset commit, Redpanda may deliver the event again.

The system records:

```text
(topic, partition, offset)
```

for processed events.

This allows duplicate deliveries to be detected before the experiment state is updated again.

On startup, persisted events are replayed to reconstruct the in-memory `ExperimentEngine` before new streaming events are processed.

---

## Tech Stack

### Data & Engineering

* Python
* PostgreSQL 16
* psycopg
* Redpanda
* Kafka-compatible event streaming
* confluent-kafka
* Docker Compose

### Statistics

* NumPy
* pandas
* SciPy
* statsmodels

### Testing

* pytest

### Infrastructure

Docker Compose currently runs:

```text
Redpanda
Redpanda Console
PostgreSQL
```

---

## Key Results

The current implementation supports:

### Real-time experiment events

Two event streams are processed:

```text
transaction.events
fraud.outcomes
```

Transactions are generated first, while fraud outcomes can arrive later.

---

### Sequential experiment monitoring

The engine continuously tracks experimental evidence as outcomes become available rather than waiting for one final batch analysis.

---

### Control vs Treatment Metrics

The state includes measures such as:

```text
Control approval rate
Treatment approval rate

Control fraud leakage
Treatment fraud leakage
```

---

### Sample Ratio Monitoring

The engine monitors **Sample Ratio Mismatch (SRM)** to detect situations where control and treatment allocation differs unexpectedly from the experimental design.

---

### Operational Guardrails

The engine includes safety checks around outcomes such as:

```text
High-value fraud
False declines
```

Experiment decisions therefore depend on more than the primary statistical metric.

---

### Durable PostgreSQL Persistence

Events and experiment snapshots are persisted before the corresponding Kafka offset is acknowledged.

This allows experiment state to survive a consumer restart.

---

### Duplicate Protection

Streaming platforms commonly provide at-least-once delivery.

If an event is redelivered, the application checks:

```text
topic
partition
offset
```

before processing it.

This provides application-level idempotency.

---

### Recovery

When the consumer starts, persisted events are replayed to reconstruct the in-memory experiment state.

The experiment therefore does not have to restart from zero after a process failure.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Vishva-23/sequential-experimentation-engine.git
cd sequential-experimentation-engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the infrastructure

```bash
docker compose up -d
```

This starts:

```text
Redpanda         localhost:19092
Redpanda Console localhost:8080
PostgreSQL       localhost:5432
```

The default PostgreSQL connection is:

```text
postgresql://experiment:experiment@localhost:5432/experiments
```

### 5. Start the experiment consumer

```bash
python -m streaming.redpanda_consumer
```

The consumer listens to:

```text
transaction.events
fraud.outcomes
```

and continuously updates the experiment state.

### 6. Produce simulated transactions

Open another terminal with the virtual environment activated:

```bash
python -m streaming.redpanda_producer \
  --count 100 \
  --rate 20 \
  --delay-scale 0.001
```

On Windows PowerShell you can run the same command on one line:

```powershell
python -m streaming.redpanda_producer --count 100 --rate 20 --delay-scale 0.001
```

### 7. Run the tests

```bash
pytest
```

### 8. Inspect Redpanda

Open:

```text
http://localhost:8080
```

to inspect topics and streamed events through Redpanda Console.

---

## Repository Structure

```text
sequential-experimentation-engine/
│
├── consumer/
│   └── experiment engine and decision logic
│
├── producer/
│   └── event models and synthetic generators
│
├── streaming/
│   ├── redpanda_producer.py
│   └── redpanda_consumer.py
│
├── persistence/
│   └── PostgreSQL persistence and recovery
│
├── simulation/
│   └── experiment simulations
│
├── tests/
│   └── automated tests
```
