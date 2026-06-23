"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { apiFetch } from "@/lib/api";
import type { AccountSummary, AccountDetail } from "@/lib/types";

function fmt(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

function pct(n: number) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

export default function Dashboard() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [equityCurves, setEquityCurves] = useState<Record<string, { date: string; equity: number }[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const accts = await apiFetch<AccountSummary[]>("/api/accounts");
        setAccounts(accts);

        const curves: Record<string, { date: string; equity: number }[]> = {};
        for (const a of accts) {
          const detail = await apiFetch<AccountDetail>(`/api/accounts/${a.name}`);
          curves[a.name] = detail.equity_curve;
        }
        setEquityCurves(curves);
      } catch (e) {
        console.error("Failed to load accounts:", e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Merge equity curves for chart
  const mergedCurve = (() => {
    const dateMap = new Map<string, Record<string, number | string>>();
    for (const [name, curve] of Object.entries(equityCurves)) {
      for (const point of curve) {
        const d = point.date.slice(0, 10);
        if (!dateMap.has(d)) dateMap.set(d, { date: d });
        dateMap.get(d)![name] = point.equity;
      }
    }
    return Array.from(dateMap.values()).sort((a, b) =>
      (a.date as string).localeCompare(b.date as string)
    );
  })();

  const colors = ["#22c55e", "#3b82f6", "#f59e0b"];

  if (loading) return <div className="text-muted-foreground p-8">Loading accounts...</div>;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {accounts.map((a, i) => (
          <Link key={`${a.name}-${i}`} href={`/accounts/${a.name}`}>
            <Card className="hover:border-primary/50 transition-colors cursor-pointer">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-base">{a.name}</CardTitle>
                  <Badge variant="outline" style={{ borderColor: colors[i] }}>
                    {a.strategy}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{fmt(a.equity)}</div>
                <div className={`text-sm ${a.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {fmt(a.total_pnl)} ({pct(a.total_return_pct)})
                </div>
                <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                  <span>{a.total_trades} trades</span>
                  <span>{(a.win_rate * 100).toFixed(0)}% WR</span>
                  <span>{a.open_positions} open</span>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {mergedCurve.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Equity Curves</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={mergedCurve}>
                <XAxis dataKey="date" tick={{ fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}K`} />
                <Tooltip formatter={(v) => fmt(Number(v))} />
                <Legend />
                {accounts.map((a, i) => (
                  <Line key={a.name} type="monotone" dataKey={a.name} stroke={colors[i]} dot={false} strokeWidth={2} name={a.strategy} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader><CardTitle>Account Summary</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Account</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead className="text-right">Equity</TableHead>
                <TableHead className="text-right">P/L</TableHead>
                <TableHead className="text-right">Return</TableHead>
                <TableHead className="text-right">Trades</TableHead>
                <TableHead className="text-right">Win Rate</TableHead>
                <TableHead className="text-right">Open</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {accounts.map((a, idx) => (
                <TableRow key={`row-${a.name}-${idx}`}>
                  <TableCell><Link href={`/accounts/${a.name}`} className="font-medium hover:underline">{a.name}</Link></TableCell>
                  <TableCell><Badge variant="secondary">{a.strategy}</Badge></TableCell>
                  <TableCell className="text-right font-mono">{fmt(a.equity)}</TableCell>
                  <TableCell className={`text-right font-mono ${a.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>{fmt(a.total_pnl)}</TableCell>
                  <TableCell className="text-right">{pct(a.total_return_pct)}</TableCell>
                  <TableCell className="text-right">{a.total_trades}</TableCell>
                  <TableCell className="text-right">{(a.win_rate * 100).toFixed(1)}%</TableCell>
                  <TableCell className="text-right">{a.open_positions}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
