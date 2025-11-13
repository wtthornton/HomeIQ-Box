# HomeIQ-Box

Open-source edge runtime for HomeIQ homes. This repository packages the on-premise services that run alongside Home Assistant, delivering real-time ingestion, inline normalization, local analytics, and optional cloud sync without exposing private data.

## Purpose
- Provide a production-ready Home Assistant companion that installs as a Home Assistant add-on or via Docker Compose
- Keep all event processing, enrichment, and dashboards local by default (Epic 31 architecture)
- Offer an auditable bridge to HomeIQ-Cloud for users who opt into AI insights

## What Lives Here
- Edge services: `websocket-ingestion`, `admin-api`, `health-dashboard`, external data fetchers, and other local-first microservices
- Shared libraries and scripts needed to operate the edge stack
- Home Assistant add-on packaging plus the open `cloud-sync` agent and privacy controls
- Documentation focused on local deployment, privacy, and contribution guidelines

## Relationship to Other Repos
- `HomeIQ` (legacy dev repo) continues to house ongoing feature work; releases are cherry-picked into this repo for stability
- `HomeIQ-Cloud` hosts proprietary SaaS, AI, and fleet services that Box can talk to over a versioned API
- `HomeIQ-Infra` manages cloud infrastructure as code (AWS-centric) that supports Cloud workloads and distribution artifacts needed by Box

## Getting Started
1. Clone the repo
   ```powershell
   git clone https://github.com/wtthornton/HomeIQ-Box.git
   cd HomeIQ-Box
   ```
2. Follow the deployment guide (coming soon) to run via Home Assistant add-on or Docker Compose
3. Enable optional cloud sync only if you want HomeIQ-Cloud insights; everything else works offline

## Next Steps
- Populate the repository with the edge services copied from `HomeIQ`
- Add docs: architecture overview, deployment instructions, privacy and security statements, contribution guide, code of conduct
- Wire CI to lint and test the local services

---
Built for transparency and trust: HomeIQ-Box lets homeowners run advanced analytics locally while staying in full control of their data.