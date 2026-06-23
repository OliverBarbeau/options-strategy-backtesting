# TradeLab Web

Next.js + shadcn/ui frontend for the TradeLab backtesting platform.

## Run

```bash
npm install
npm run dev
```

Requires the API running on port 8000:
```bash
cd ..
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

## Pages

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | `app/page.tsx` | Dashboard: account cards, combined equity curve, summary table |
| `/backtest` | `app/backtest/page.tsx` | Parameter form, SSE progress, metrics, trade log |
| `/accounts` | `app/accounts/page.tsx` | Grid of all simulation accounts |
| `/accounts/[name]` | `app/accounts/[name]/page.tsx` | Tabs: overview, positions w/ MTM, trade history |
| `/broker` | `app/broker/page.tsx` | Schwab connection, spread scanner |

## Key Files

- `src/lib/api.ts` — Fetch wrapper + EventSource helper for SSE
- `src/lib/types.ts` — TypeScript types matching FastAPI Pydantic schemas
- `src/components/ui/*` — shadcn/ui components (auto-generated)

## Tech Stack

- Next.js 16 with App Router
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui (neutral theme)
- Recharts for equity curves and P/L charts

## Adding shadcn Components

```bash
npx shadcn@latest add <component>
```

Example: `npx shadcn@latest add dialog tooltip` to add more components.

## Environment

`NEXT_PUBLIC_API_URL` — API base URL (defaults to `http://localhost:8000`)

## Build

```bash
npm run build    # Production build
npm run start    # Run production build
```
