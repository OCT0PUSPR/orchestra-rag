# Nimbus Engineering Handbook

This handbook describes how the engineering organization at Nimbus Robotics
works.

## Source control and code review

All code lives in a single monorepo hosted on an internal Git server. Every
change goes through a pull request, and every pull request requires at least one
approving review before it can merge. Changes to safety-critical firmware
require two approvals, one of which must come from a member of the Controls team.

## Testing requirements

Every service must ship with automated tests. The continuous-integration system
runs the full test suite on every pull request, and a change cannot merge if any
test fails. New features are expected to include both unit tests and at least one
integration test. The target line-coverage threshold for the Conductor codebase
is 80 percent.

## On-call and incident response

Each team runs a weekly on-call rotation. When an incident is declared, the
on-call engineer becomes the incident commander until the issue is resolved.
After every incident the team writes a blameless postmortem within three business
days and shares it company-wide.

## Release cadence

The Conductor platform ships on a weekly release train every Tuesday. Atlas-7
firmware ships on a slower monthly cadence because each firmware change must pass
a hardware-in-the-loop regression suite before it can be promoted.

## Programming languages

The approved production languages at Nimbus are Rust for real-time and
performance-critical systems, Python for services and tooling, and TypeScript
for the dashboard front end. Introducing a new production language requires sign
off from the engineering leadership team.
