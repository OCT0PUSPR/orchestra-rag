# Conductor — Fleet Coordination Platform

**Conductor** is the cloud platform that coordinates fleets of Atlas-7 robots.
It assigns tasks, plans traffic, and exposes a dashboard and an HTTP API.

## Architecture

Conductor runs as a set of services in Kubernetes. The core scheduler is written
in Rust for low-latency task assignment, while the dashboard and public API are
written in Python using FastAPI. Robot telemetry streams in over MQTT and is
stored in a time-series database.

## Traffic management

Conductor prevents collisions and deadlocks using a reservation system: before a
robot enters an intersection or a narrow aisle, it reserves the cells it needs
from the scheduler. If two robots request overlapping cells, the one with the
higher-priority task wins and the other waits. Priorities are derived from task
deadlines, so late orders automatically get right-of-way.

## API

The Conductor public API is REST over HTTPS. Authentication uses API keys passed
in the `Authorization` header as a bearer token. Key endpoints include
`POST /v1/tasks` to enqueue a picking task, `GET /v1/fleet` to list robot
status, and `GET /v1/tasks/{id}` to check a task's progress. The API is
rate-limited to 100 requests per second per API key.

## Deployment options

Conductor is offered as a managed cloud service hosted in Lisbon, and also as a
self-hosted on-premises bundle for customers with strict data-residency
requirements. The on-premises bundle ships as a Helm chart.
