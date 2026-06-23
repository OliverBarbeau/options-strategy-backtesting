"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { apiFetch, apiStream } from "@/lib/api";
import type { BacktestResponse, Strategy } from "@/lib/types";
import { DrawdownChart } from "@/components/charts/drawdown-chart";
import { PnLDistribution } from "@/components/charts/pnl-distribution";
import { TickerBarChart } from "@/components/charts/ticker-bar-chart";
import { MonthlyHeatmap } from "@/components/charts/monthly-heatmap";

const TICKER_OPTIONS = [
  "META", "AVGO", "MSFT", "GOOG", "NVDA", "CAT", "AAPL",
  "AMD", "QQQ", "SPY", "IWM", "JPM", "WMT", "HD", "XLE",
];

function fmt(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export default function BacktestPage() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [strategy, setStrategy] = useState("pullback");
  const [tickers, setTickers] = useState<string[]>(["META", "AVGO", "MSFT", "GOOG", "NVDA", "CAT"]);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2026-04-04");
  const [buffer, setBuffer] = useState(0.10);
  const [spreadPct, setSpreadPct] = useState(0.02);
  const [dteOpen, setDteOpen] = useState(30);
  const [dteClose, setDteClose] = useState(14);
  const [pullbackThreshold, setPullbackThreshold] = useState(0.03);
  const [maxContracts, setMaxContracts] = useState(10);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [tradeFilter, setTradeFilter] = useState<"all" | "winners" | "losers">("all");

  useEffect(() => {
    apiFetch<{ strategies: Strategy[] }>("/api/strategies").then((d) =>
      setStrategies(d.strategies)
    );
  }, []);

  // Update defaults when strategy changes
  useEffect(() => {
    const s = strategies.find((x) => x.id === strategy);
    if (s) {
      setBuffer(s.defaults.buffer ?? 0.10);
      setSpreadPct(s.defaults.spread_pct ?? 0.02);
      setDteOpen(s.defaults.dte_open ?? 30);
      setDteClose(s.defaults.dte_close ?? 14);
      setPullbackThreshold(s.defaults.pullback_threshold ?? 0.03);
    }
  }, [strategy, strategies]);

  function toggleTicker(t: string) {
    setTickers((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  }

  async function runBacktest() {
    setRunning(true);
    setProgress(0);
    setResult(null);

    try {
      const { id } = await apiFetch<{ id: string }>("/api/backtests", {
        method: "POST",
        body: JSON.stringify({
          strategy, tickers, start_date: startDate, end_date: endDate,
          buffer, spread_pct: spreadPct, dte_open: dteOpen, dte_close: dteClose,
          pullback_threshold: pullbackThreshold, max_contracts: maxContracts,
        }),
      });

      apiStream(`/api/backtests/${id}/stream`, (event, data) => {
        if (event === "progress") {
          const d = JSON.parse(data);
          setProgress(d.progress * 100);
          setProgressMsg(d.message);
        } else if (event === "complete") {
          const d = JSON.parse(data);
          setResult(d);
          setRunning(false);
          setProgress(100);
        } else if (event === "error") {
          setProgressMsg(`Error: ${data}`);
          setRunning(false);
        }
      });
    } catch (e) {
      setProgressMsg(`Failed: ${e}`);
      setRunning(false);
    }
  }

  const filteredTrades = result?.trade_log.filter((t) => {
    if (tradeFilter === "winners") return t.winner;
    if (tradeFilter === "losers") return !t.winner;
    return true;
  }) ?? [];

  const m = result?.metrics;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Backtest Runner</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Parameter Form */}
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Strategy</Label>
              <Select value={strategy} onValueChange={(v) => v && setStrategy(v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {strategies.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {strategies.find((s) => s.id === strategy)?.description}
              </p>
            </div>

            <div>
              <Label>Tickers</Label>
              <div className="flex flex-wrap gap-1 mt-1">
                {TICKER_OPTIONS.map((t) => (
                  <Badge
                    key={t}
                    variant={tickers.includes(t) ? "default" : "outline"}
                    className="cursor-pointer text-xs"
                    onClick={() => toggleTicker(t)}
                  >
                    {t}
                  </Badge>
                ))}
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>Start Date</Label>
                <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
              </div>
              <div>
                <Label>End Date</Label>
                <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
              </div>
              <div>
                <Label>Buffer %</Label>
                <Input type="number" step="0.01" value={buffer} onChange={(e) => setBuffer(+e.target.value)} />
              </div>
              <div>
                <Label>Spread %</Label>
                <Input type="number" step="0.01" value={spreadPct} onChange={(e) => setSpreadPct(+e.target.value)} />
              </div>
              <div>
                <Label>DTE Open</Label>
                <Input type="number" value={dteOpen} onChange={(e) => setDteOpen(+e.target.value)} />
              </div>
              <div>
                <Label>DTE Close</Label>
                <Input type="number" value={dteClose} onChange={(e) => setDteClose(+e.target.value)} />
              </div>
              <div>
                <Label>Pullback %</Label>
                <Input type="number" step="0.01" value={pullbackThreshold} onChange={(e) => setPullbackThreshold(+e.target.value)} />
              </div>
              <div>
                <Label>Max Contracts</Label>
                <Input type="number" value={maxContracts} onChange={(e) => setMaxContracts(+e.target.value)} />
              </div>
            </div>

            <Button onClick={runBacktest} disabled={running || tickers.length === 0} className="w-full">
              {running ? "Running..." : "Run Backtest"}
            </Button>

            {running && (
              <div>
                <Progress value={progress} />
                <p className="text-xs text-muted-foreground mt-1">{progressMsg}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {m && (
            <>
              {/* Metrics Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: "Total P/L", value: fmt(m.total_pnl), color: m.total_pnl >= 0 },
                  { label: "Win Rate", value: `${(m.win_rate * 100).toFixed(1)}%` },
                  { label: "Trades", value: `${m.winners}W / ${m.losers}L` },
                  { label: "Max Drawdown", value: `${(m.max_drawdown_pct * 100).toFixed(1)}%` },
                  { label: "Avg $/Trade", value: fmt(m.avg_pnl) },
                  { label: "Avg Winner", value: fmt(m.avg_winner) },
                  { label: "Avg Loser", value: fmt(m.avg_loser) },
                  { label: "Duration", value: `${result!.duration_seconds.toFixed(1)}s` },
                ].map((card) => (
                  <Card key={card.label}>
                    <CardContent className="pt-4 pb-3">
                      <div className="text-xs text-muted-foreground">{card.label}</div>
                      <div className={`text-lg font-bold ${card.color === true ? "text-green-500" : card.color === false ? "text-red-500" : ""}`}>
                        {card.value}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Equity Curve + Drawdown (side by side) */}
              {result!.equity_curve.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader><CardTitle className="text-sm">Cumulative P/L</CardTitle></CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={220}>
                        <AreaChart data={result!.equity_curve}>
                          <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v?.slice(5, 10)} />
                          <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
                          <Tooltip formatter={(v) => fmt(Number(v))} />
                          <Area type="monotone" dataKey="equity" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader><CardTitle className="text-sm">Drawdown</CardTitle></CardHeader>
                    <CardContent>
                      <DrawdownChart data={result!.equity_curve} height={220} />
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Per-Ticker P/L + P/L Distribution (side by side) */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {result!.ticker_results.length > 1 && (
                  <Card>
                    <CardHeader><CardTitle className="text-sm">P/L by Ticker</CardTitle></CardHeader>
                    <CardContent>
                      <TickerBarChart data={result!.ticker_results} height={220} />
                    </CardContent>
                  </Card>
                )}
                <Card>
                  <CardHeader><CardTitle className="text-sm">P/L Distribution</CardTitle></CardHeader>
                  <CardContent>
                    <PnLDistribution trades={result!.trade_log} height={220} />
                  </CardContent>
                </Card>
              </div>

              {/* Monthly Returns Heatmap */}
              <Card>
                <CardHeader><CardTitle className="text-sm">Monthly Returns</CardTitle></CardHeader>
                <CardContent>
                  <MonthlyHeatmap trades={result!.trade_log.map(t => ({ date: t.date, pnl: t.pnl }))} />
                </CardContent>
              </Card>

              {/* Trade Log */}
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <CardTitle className="text-sm">Trade Log ({filteredTrades.length})</CardTitle>
                    <div className="flex gap-1">
                      {(["all", "winners", "losers"] as const).map((f) => (
                        <Badge key={f} variant={tradeFilter === f ? "default" : "outline"} className="cursor-pointer text-xs" onClick={() => setTradeFilter(f)}>
                          {f}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="max-h-96 overflow-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Date</TableHead>
                          <TableHead>Ticker</TableHead>
                          <TableHead className="text-right">Entry</TableHead>
                          <TableHead className="text-right">Exit</TableHead>
                          <TableHead className="text-right">P/L</TableHead>
                          <TableHead className="text-right">Contracts</TableHead>
                          <TableHead className="text-right">HV</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredTrades.slice(0, 200).map((t, i) => (
                          <TableRow key={i}>
                            <TableCell className="text-xs">{t.date.slice(0, 10)}</TableCell>
                            <TableCell><Badge variant="outline" className="text-xs">{t.ticker}</Badge></TableCell>
                            <TableCell className="text-right font-mono text-xs">${t.entry_price.toFixed(2)}</TableCell>
                            <TableCell className="text-right font-mono text-xs">${t.exit_price.toFixed(2)}</TableCell>
                            <TableCell className={`text-right font-mono text-xs ${t.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                              {fmt(t.pnl)}
                            </TableCell>
                            <TableCell className="text-right text-xs">{t.contracts}</TableCell>
                            <TableCell className="text-right text-xs">{(t.sigma * 100).toFixed(0)}%</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </>
          )}

          {!m && !running && (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                Configure parameters and click Run Backtest to see results.
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
