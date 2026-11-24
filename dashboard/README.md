# dashboard/: Web operator dashboard

- **Owner:** Ario Bashiri (CE)
- **Status:** planned

## Purpose

A React or Next.js web application for traffic operators. It shows live lane queues, signal phases, emergency vehicle tracking, CO₂ KPIs, forecasts and the LLM briefings and alerts.

## Scope

**Belongs here:** the frontend application, its components, state management, map views, charts, the WebSocket/REST client, and eslint + prettier configuration.

**Does not belong here:** backend services and data storage (`backend/`), model logic (`ai/`), the mobile app (owner TBD).

## Planned sub-structure

```
dashboard/
├── package.json, eslint and prettier config
├── src/        # pages/app, components, hooks, api client
├── public/
└── tests/
```

## Interfaces

- **Consumes:** the backend REST/WebSocket API (to be defined by Ario), which carries data in the formats of [`shared/contracts/schemas/`](../shared/contracts/schemas/).
- **Produces:** operator actions, if any (e.g. manual overrides). These must go through the backend and respect the signal safety limits.

## Setup, run, test

Requires Node.js LTS. Commands (`npm install`, `npm run dev`, `npm run lint`, `npm test`) will be added when the app is scaffolded. CI runs the dashboard job only once `dashboard/package.json` exists.
