"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DrawdownChart } from "@/components/charts/drawdown-chart";
import { MonthlyHeatmap } from "@/components/charts/monthly-heatmap";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { apiFetch } from "@/lib/api";
import type { AccountDetail, MtmPosition, PositionDetail } from "@/lib/types";

function fmt(n: number) {
  return n.toLocaleString("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  });
}
function fmtPrecise(n: number) {
  return n.toLocaleString("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 2,
  });
}
function pct(n: number) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

// Merge position static data with MTM live data
interface MergedPosition {
  pos: PositionDetail;
  mtm: MtmPosition | null;
  // Computed fields
  spreadWidth: number;
  maxLoss: number;
  initBuffer: number;
  currentBuffer: number | null;
  underlyingChange: number | null;
  creditPotential: number;
  unrealizedPnl: number | null;
  returnOnRisk: number | null;
  daysHeld: number;
  dteRemaining: number | null;
  expiry: string;
}

function mergePositions(positions: PositionDetail[], mtmData: MtmPosition[]): MergedPosition[] {
  const mtmMap = new Map(mtmData.map((m) => [m.id, m]));

  return positions.map((pos) => {
    const mtm = mtmMap.get(pos.id) || null;
    const spreadWidth = (pos.short_strike - pos.long_strike) * 100 * pos.contracts;
    const maxLoss = spreadWidth - pos.credit_received;
    const initBuffer = ((pos.entry_price - pos.short_strike) / pos.entry_price) * 100;
    const currentBuffer = mtm ? ((mtm.current_price - pos.short_strike) / mtm.current_price) * 100 : null;
    const underlyingChange = mtm ? ((mtm.current_price - pos.entry_price) / pos.entry_price) * 100 : null;
    const creditPotential = maxLoss > 0 ? (pos.credit_received / maxLoss) * 100 : 0;
    const unrealizedPnl = mtm ? pos.credit_received - mtm.close_cost : null;
    const returnOnRisk = unrealizedPnl !== null && maxLoss > 0 ? (unrealizedPnl / maxLoss) * 100 : null;

    const entryDate = new Date(pos.entry_date);
    const today = new Date();
    const daysHeld = Math.floor((today.getTime() - entryDate.getTime()) / 86400000);
    const dteRemaining = mtm ? mtm.dte_remaining : null;

    // Expiry: use notes field (actual chain expiry) if set,
    // otherwise estimate from DTE remaining or close_target + 14 days
    let expiry = "";
    if (pos.notes && pos.notes.match(/^\d{4}-\d{2}-\d{2}/)) {
      expiry = pos.notes.slice(0, 10);
    } else if (dteRemaining !== null) {
      const exp = new Date();
      exp.setDate(exp.getDate() + dteRemaining);
      expiry = exp.toISOString().slice(0, 10);
    } else {
      // close_target is 14 DTE checkpoint, so expiry ≈ close_target + 14 days
      const ct = new Date(pos.close_target_date);
      ct.setDate(ct.getDate() + 14);
      expiry = ct.toISOString().slice(0, 10);
    }

    return {
      pos, mtm, spreadWidth, maxLoss, initBuffer, currentBuffer,
      underlyingChange, creditPotential, unrealizedPnl, returnOnRisk,
      daysHeld, dteRemaining, expiry,
    };
  });
}

export default function AccountDetailPage() {
  const params = useParams<{ name: string }>();
  const [detail, setDetail] = useState<AccountDetail | null>(null);
  const [mtm, setMtm] = useState<MtmPosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [mtmLoading, setMtmLoading] = useState(false);
  const [mtmError, setMtmError] = useState("");
  const [lastMtmTime, setLastMtmTime] = useState<string>("");

  const loadMtm = useCallback(async () => {
    setMtmLoading(true);
    setMtmError("");
    try {
      const data = await apiFetch<MtmPosition[]>(`/api/accounts/${params.name}/mtm`);
      setMtm(data);
      setLastMtmTime(new Date().toLocaleTimeString());
    } catch (e) {
      setMtmError(String(e));
    } finally {
      setMtmLoading(false);
    }
  }, [params.name]);

  useEffect(() => {
    async function load() {
      try {
        const d = await apiFetch<AccountDetail>(`/api/accounts/${params.name}`);
        setDetail(d);
        // Auto-load MTM if there are open positions
        if (d.positions.length > 0) {
          try {
            const data = await apiFetch<MtmPosition[]>(`/api/accounts/${params.name}/mtm`);
            setMtm(data);
            setLastMtmTime(new Date().toLocaleTimeString());
          } catch { /* MTM may fail if no market data, that's ok */ }
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [params.name]);

  if (loading || !detail) return <div className="text-muted-foreground p-8">Loading...</div>;

  const s = detail.summary;
  const peakEquity = detail.equity_curve.length > 0
    ? Math.max(...detail.equity_curve.map((e) => e.equity))
    : s.equity;
  const currentDD = ((s.equity - peakEquity) / peakEquity * 100);

  const merged = mergePositions(detail.positions, mtm);
  const totalUnrealized = merged.reduce((sum, m) => sum + (m.unrealizedPnl ?? 0), 0);
  const totalCollateral = merged.reduce((sum, m) => sum + m.pos.collateral, 0);
  const totalCredit = merged.reduce((sum, m) => sum + m.pos.credit_received, 0);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">{s.name}</h2>
          <Badge variant="secondary">{s.strategy}</Badge>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold">{fmt(s.equity)}</div>
          <div className={`text-sm ${s.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
            {fmt(s.total_pnl)} ({pct(s.total_return_pct)})
          </div>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        {[
          { label: "Balance", value: fmt(s.balance) },
          { label: "Locked", value: fmt(s.locked) },
          { label: "Trades", value: `${s.total_trades}` },
          { label: "Win Rate", value: `${(s.win_rate * 100).toFixed(1)}%` },
          { label: "Open Positions", value: `${s.open_positions}` },
          { label: "Current DD", value: `${currentDD.toFixed(1)}%` },
        ].map((m) => (
          <Card key={m.label}>
            <CardContent className="pt-3 pb-2">
              <div className="text-xs text-muted-foreground">{m.label}</div>
              <div className="text-lg font-bold">{m.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="positions">
        <TabsList>
          <TabsTrigger value="positions">Positions ({s.open_positions})</TabsTrigger>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trades">Trade History</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        {/* Positions Tab — now the default, options-specific layout */}
        <TabsContent value="positions" className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-lg font-semibold">Put Credit Spreads</h3>
              {lastMtmTime && (
                <span className="text-xs text-muted-foreground">
                  Priced at {lastMtmTime}
                </span>
              )}
            </div>
            <Button onClick={loadMtm} disabled={mtmLoading} variant="outline" size="sm">
              {mtmLoading ? "Refreshing..." : "Refresh Prices"}
            </Button>
          </div>

          {mtmError && (
            <p className="text-xs text-red-500">{mtmError}</p>
          )}

          {merged.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                No open positions
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Opened</TableHead>
                      <TableHead className="text-center">Strikes</TableHead>
                      <TableHead className="text-right">Qty</TableHead>
                      <TableHead className="text-right">Entry</TableHead>
                      <TableHead className="text-right">Current</TableHead>
                      <TableHead className="text-right">Chg</TableHead>
                      <TableHead className="text-right">Buffer</TableHead>
                      <TableHead className="text-right">Credit</TableHead>
                      <TableHead className="text-right">Max Loss</TableHead>
                      <TableHead className="text-right">Cr. Pot.</TableHead>
                      <TableHead className="text-right">Unreal P/L</TableHead>
                      <TableHead className="text-right">Ret/Risk</TableHead>
                      <TableHead className="text-right">DTE</TableHead>
                      <TableHead>Expiry</TableHead>
                      <TableHead>Close By</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {merged.map((m) => {
                      const hasLive = m.mtm !== null;
                      const bufferDanger = (m.currentBuffer ?? m.initBuffer) < 5;
                      const bufferWarning = (m.currentBuffer ?? m.initBuffer) < 8;
                      return (
                        <TableRow key={m.pos.id} className={bufferDanger ? "bg-red-500/10" : bufferWarning ? "bg-yellow-500/5" : ""}>
                          <TableCell className="font-bold">{m.pos.ticker}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">{m.pos.entry_date.slice(0, 10)}</TableCell>
                          <TableCell className="text-center font-mono text-sm">
                            <span className="text-muted-foreground">sell</span> {m.pos.short_strike.toFixed(0)}
                            {" / "}
                            <span className="text-muted-foreground">buy</span> {m.pos.long_strike.toFixed(0)}
                          </TableCell>
                          <TableCell className="text-right">{m.pos.contracts}x</TableCell>
                          <TableCell className="text-right font-mono text-sm">{fmtPrecise(m.pos.entry_price)}</TableCell>
                          <TableCell className="text-right font-mono text-sm">
                            {hasLive ? fmtPrecise(m.mtm!.current_price) : <span className="text-muted-foreground">--</span>}
                          </TableCell>
                          <TableCell className={`text-right text-sm ${(m.underlyingChange ?? 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {m.underlyingChange !== null ? pct(m.underlyingChange) : "--"}
                          </TableCell>
                          <TableCell className={`text-right text-sm font-medium ${bufferDanger ? "text-red-500" : bufferWarning ? "text-yellow-500" : "text-green-500"}`}>
                            {(m.currentBuffer ?? m.initBuffer).toFixed(1)}%
                          </TableCell>
                          <TableCell className="text-right font-mono text-sm text-green-500">{fmt(m.pos.credit_received)}</TableCell>
                          <TableCell className="text-right font-mono text-sm text-red-500">{fmt(m.maxLoss)}</TableCell>
                          <TableCell className="text-right text-sm">{m.creditPotential.toFixed(0)}%</TableCell>
                          <TableCell className={`text-right font-mono text-sm font-bold ${(m.unrealizedPnl ?? 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {m.unrealizedPnl !== null ? fmt(m.unrealizedPnl) : "--"}
                          </TableCell>
                          <TableCell className={`text-right text-sm ${(m.returnOnRisk ?? 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {m.returnOnRisk !== null ? `${m.returnOnRisk.toFixed(1)}%` : "--"}
                          </TableCell>
                          <TableCell className="text-right text-sm">{m.dteRemaining ?? "--"}</TableCell>
                          <TableCell className="text-sm">{m.expiry}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">{m.pos.close_target_date.slice(0, 10)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              {/* Summary row */}
              <div className="flex gap-6 text-sm px-2">
                <div>
                  <span className="text-muted-foreground">Total collateral: </span>
                  <span className="font-mono font-medium">{fmt(totalCollateral)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Total credit: </span>
                  <span className="font-mono font-medium text-green-500">{fmt(totalCredit)}</span>
                </div>
                {totalUnrealized !== 0 && (
                  <div>
                    <span className="text-muted-foreground">Total unrealized: </span>
                    <span className={`font-mono font-bold ${totalUnrealized >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {fmt(totalUnrealized)}
                    </span>
                  </div>
                )}
                <div>
                  <span className="text-muted-foreground">Deployment: </span>
                  <span className="font-medium">{s.equity > 0 ? ((s.locked / s.equity) * 100).toFixed(0) : 0}%</span>
                </div>
              </div>
            </>
          )}
        </TabsContent>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          {detail.equity_curve.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader><CardTitle className="text-sm">Equity Curve</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={detail.equity_curve}>
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5, 10)} />
                      <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}K`} />
                      <Tooltip formatter={(v) => fmt(Number(v))} />
                      <Area type="monotone" dataKey="equity" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle className="text-sm">Drawdown</CardTitle></CardHeader>
                <CardContent>
                  <DrawdownChart data={detail.equity_curve} height={250} />
                </CardContent>
              </Card>
            </div>
          )}
          {detail.recent_trades.length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-sm">Monthly Returns</CardTitle></CardHeader>
              <CardContent>
                <MonthlyHeatmap trades={detail.recent_trades.map(t => ({ date: t.entry_date, pnl: t.pnl }))} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Trade History Tab */}
        <TabsContent value="trades">
          <Card>
            <CardHeader><CardTitle className="text-sm">Recent Trades</CardTitle></CardHeader>
            <CardContent>
              <div className="max-h-[500px] overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead></TableHead>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Entry</TableHead>
                      <TableHead>Exit</TableHead>
                      <TableHead className="text-right">Entry $</TableHead>
                      <TableHead className="text-right">Exit $</TableHead>
                      <TableHead className="text-right">P/L</TableHead>
                      <TableHead>Reason</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {detail.recent_trades.slice().reverse().map((t) => (
                      <TableRow key={t.id}>
                        <TableCell>
                          <Badge variant={t.winner ? "default" : "destructive"} className="text-xs">
                            {t.winner ? "W" : "L"}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-medium">{t.ticker}</TableCell>
                        <TableCell className="text-xs">{t.entry_date.slice(0, 10)}</TableCell>
                        <TableCell className="text-xs">{t.exit_date.slice(0, 10)}</TableCell>
                        <TableCell className="text-right font-mono text-xs">${t.entry_price.toFixed(2)}</TableCell>
                        <TableCell className="text-right font-mono text-xs">${t.exit_price.toFixed(2)}</TableCell>
                        <TableCell className={`text-right font-mono ${t.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {fmt(t.pnl)}
                        </TableCell>
                        <TableCell><Badge variant="outline" className="text-xs">{t.exit_reason}</Badge></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Config Tab */}
        <TabsContent value="config">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="text-sm">Test Configuration & Parameters</CardTitle>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    navigator.clipboard.writeText(detail.config_summary);
                    const btn = document.activeElement as HTMLButtonElement;
                    if (btn) {
                      const orig = btn.textContent;
                      btn.textContent = "Copied!";
                      setTimeout(() => { btn.textContent = orig; }, 2000);
                    }
                  }}
                >
                  Copy to Clipboard
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <pre className="text-sm font-mono whitespace-pre-wrap bg-muted/50 rounded-md p-4 leading-relaxed">
                {detail.config_summary || "No config data available"}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
