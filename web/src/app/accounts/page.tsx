"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { apiFetch } from "@/lib/api";
import type { AccountSummary } from "@/lib/types";

function fmt(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

// localStorage helpers for starred/archived state
function getStarred(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    return new Set(JSON.parse(localStorage.getItem("tradelab_starred") || "[]"));
  } catch { return new Set(); }
}
function setStarred(s: Set<string>) {
  localStorage.setItem("tradelab_starred", JSON.stringify([...s]));
}
function getArchived(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    return new Set(JSON.parse(localStorage.getItem("tradelab_archived") || "[]"));
  } catch { return new Set(); }
}
function setArchived(s: Set<string>) {
  localStorage.setItem("tradelab_archived", JSON.stringify([...s]));
}

function AccountCard({
  a,
  starred,
  archived,
  onToggleStar,
  onToggleArchive,
}: {
  a: AccountSummary;
  starred: boolean;
  archived: boolean;
  onToggleStar: () => void;
  onToggleArchive: () => void;
}) {
  return (
    <Card className={`transition-colors ${starred ? "border-yellow-500/50" : "hover:border-primary/50"}`}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <Link href={`/accounts/${a.name}`} className="hover:underline">
            <CardTitle className="text-base">{a.name}</CardTitle>
          </Link>
          <div className="flex gap-1 items-center">
            <Badge variant="secondary" className="text-xs">{a.strategy}</Badge>
            <button
              onClick={onToggleStar}
              className="text-lg hover:scale-110 transition-transform ml-1"
              title={starred ? "Unstar" : "Star"}
            >
              {starred ? "\u2605" : "\u2606"}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <Link href={`/accounts/${a.name}`}>
          <div className="text-2xl font-bold">{fmt(a.equity)}</div>
          <div className={`text-sm font-medium ${a.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
            {fmt(a.total_pnl)} ({a.total_return_pct >= 0 ? "+" : ""}{a.total_return_pct.toFixed(1)}%)
          </div>
          <div className="flex gap-3 mt-1 text-xs text-muted-foreground">
            <span>{a.total_trades} trades</span>
            <span>{(a.win_rate * 100).toFixed(0)}% WR</span>
            <span>{a.open_positions} open</span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {fmt(a.balance)} avail | {fmt(a.locked)} locked
          </div>
        </Link>
        <div className="flex justify-between items-center pt-1">
          <span className="text-xs text-muted-foreground">
            Updated: {a.last_advanced_date?.slice(0, 10)}
          </span>
          <button
            onClick={onToggleArchive}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {archived ? "Unarchive" : "Archive"}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [starred, setStarredState] = useState<Set<string>>(new Set());
  const [archived, setArchivedState] = useState<Set<string>>(new Set());
  const [advancing, setAdvancing] = useState(false);
  const [advanceMsg, setAdvanceMsg] = useState("");

  useEffect(() => {
    apiFetch<AccountSummary[]>("/api/accounts").then(setAccounts);
    setStarredState(getStarred());
    setArchivedState(getArchived());
  }, []);

  const toggleStar = useCallback((name: string) => {
    setStarredState((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      setStarred(next);
      return next;
    });
  }, []);

  const toggleArchive = useCallback((name: string) => {
    setArchivedState((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      setArchived(next);
      return next;
    });
  }, []);

  async function advanceAll() {
    setAdvancing(true);
    setAdvanceMsg("Advancing all accounts...");
    try {
      const result = await apiFetch<{ advanced: number; results: { name: string; status: string; equity?: number }[] }>(
        "/api/accounts/advance-all",
        { method: "POST", body: JSON.stringify({}) }
      );
      setAdvanceMsg(`Advanced ${result.advanced} accounts`);
      // Refresh account list
      const refreshed = await apiFetch<AccountSummary[]>("/api/accounts");
      setAccounts(refreshed);
    } catch (e) {
      setAdvanceMsg(`Error: ${e}`);
    } finally {
      setAdvancing(false);
      setTimeout(() => setAdvanceMsg(""), 5000);
    }
  }

  // Categorize accounts
  const starredAccounts = accounts.filter((a) => starred.has(a.name) && !archived.has(a.name));
  const liveAccounts = accounts.filter(
    (a) => !starred.has(a.name) && !archived.has(a.name) && a.account_type === "live"
  );
  const backtestAccounts = accounts.filter(
    (a) => !starred.has(a.name) && !archived.has(a.name) && a.account_type === "backtest"
  );
  const archivedAccounts = accounts.filter((a) => archived.has(a.name));

  function Section({
    title,
    subtitle,
    items,
    defaultCollapsed = false,
  }: {
    title: string;
    subtitle?: string;
    items: AccountSummary[];
    defaultCollapsed?: boolean;
  }) {
    const [collapsed, setCollapsed] = useState(defaultCollapsed);
    if (items.length === 0) return null;
    return (
      <section>
        <div className="flex justify-between items-center mb-2">
          <div>
            <h2 className="text-xl font-bold">{title}</h2>
            {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            {collapsed ? `Show (${items.length})` : "Collapse"}
          </button>
        </div>
        {!collapsed && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {items.map((a) => (
              <AccountCard
                key={a.name}
                a={a}
                starred={starred.has(a.name)}
                archived={archived.has(a.name)}
                onToggleStar={() => toggleStar(a.name)}
                onToggleArchive={() => toggleArchive(a.name)}
              />
            ))}
          </div>
        )}
      </section>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Accounts</h1>
        <div className="flex items-center gap-3">
          {advanceMsg && (
            <span className="text-sm text-muted-foreground">{advanceMsg}</span>
          )}
          <Button onClick={advanceAll} disabled={advancing} variant="outline" size="sm">
            {advancing ? "Advancing..." : "Advance All to Today"}
          </Button>
        </div>
      </div>

      <Section
        title={"\u2605 Starred"}
        subtitle="Your selected strategies"
        items={starredAccounts}
      />

      <Section
        title="Live Portfolios"
        subtitle="Forward-running simulated portfolios"
        items={liveAccounts}
      />

      <Section
        title="Historical Backtests"
        subtitle="Strategy performance on historical data"
        items={backtestAccounts}
      />

      {archivedAccounts.length > 0 && (
        <>
          <Separator />
          <Section
            title="Archived"
            items={archivedAccounts}
            defaultCollapsed={true}
          />
        </>
      )}
    </div>
  );
}
