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
import { apiFetch } from "@/lib/api";
import type { AccountSummary } from "@/lib/types";

function fmt(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export default function HistoryPage() {
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch<AccountSummary[]>("/api/accounts")
      .then((all) => {
        // Show both backtest accounts and historical portfolio sims
        const historical = all.filter(
          (a) => a.account_type === "backtest" || a.name.startsWith("bt_") || a.name.startsWith("portfolio_theta")
        );
        setAccounts(historical);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-muted-foreground p-8">Loading...</div>;

  // Sort by total return descending
  const sorted = [...accounts].sort((a, b) => b.total_return_pct - a.total_return_pct);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Historical Backtests</h1>
        <p className="text-sm text-muted-foreground">
          Strategy performance on real Theta Data options, compounded across multiple years
        </p>
      </div>

      {sorted.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No backtest results yet. Run experiments to generate historical data.
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Summary cards for top 3 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {sorted.slice(0, 3).map((a, i) => (
              <Link key={a.name} href={`/accounts/${a.name}`}>
                <Card className={`cursor-pointer transition-colors ${i === 0 ? "border-yellow-500/50" : "hover:border-primary/50"}`}>
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-sm">{a.name}</CardTitle>
                      {i === 0 && <Badge className="bg-yellow-600 text-xs">Best</Badge>}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{fmt(a.equity)}</div>
                    <div className={`text-sm font-medium ${a.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {fmt(a.total_pnl)} ({a.total_return_pct >= 0 ? "+" : ""}{a.total_return_pct.toFixed(1)}%)
                    </div>
                    <div className="flex gap-3 mt-1 text-xs text-muted-foreground">
                      <span>{a.total_trades} trades</span>
                      <span>{(a.win_rate * 100).toFixed(0)}% WR</span>
                      <span>from ${a.starting_capital.toLocaleString()}</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>

          {/* Full results table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">All Backtest Results</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Rank</TableHead>
                    <TableHead>Config</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead className="text-right">Start $</TableHead>
                    <TableHead className="text-right">Final $</TableHead>
                    <TableHead className="text-right">Return</TableHead>
                    <TableHead className="text-right">Trades</TableHead>
                    <TableHead className="text-right">Win Rate</TableHead>
                    <TableHead>Period</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sorted.map((a, i) => (
                    <TableRow key={a.name} className={i === 0 ? "bg-yellow-500/5" : ""}>
                      <TableCell className="font-medium">{i + 1}</TableCell>
                      <TableCell>
                        <Link href={`/accounts/${a.name}`} className="font-medium hover:underline">
                          {a.name}
                        </Link>
                      </TableCell>
                      <TableCell><Badge variant="secondary" className="text-xs">{a.strategy}</Badge></TableCell>
                      <TableCell className="text-right font-mono">{fmt(a.starting_capital)}</TableCell>
                      <TableCell className="text-right font-mono font-bold">{fmt(a.equity)}</TableCell>
                      <TableCell className={`text-right font-mono font-bold ${a.total_return_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                        {a.total_return_pct >= 0 ? "+" : ""}{a.total_return_pct.toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right">{a.total_trades}</TableCell>
                      <TableCell className="text-right">{(a.win_rate * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {a.last_advanced_date?.slice(0, 10)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
