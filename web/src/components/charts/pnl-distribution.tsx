"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

interface Trade {
  pnl: number;
  winner: boolean;
}

interface PnLDistributionProps {
  trades: Trade[];
  height?: number;
  bins?: number;
}

export function PnLDistribution({
  trades,
  height = 220,
  bins = 20,
}: PnLDistributionProps) {
  if (!trades || trades.length < 3) return null;

  const pnls = trades.map((t) => t.pnl);
  const min = Math.min(...pnls);
  const max = Math.max(...pnls);
  const range = max - min || 1;
  const binWidth = range / bins;

  // Build histogram
  const histogram: { range: string; count: number; mid: number }[] = [];
  for (let i = 0; i < bins; i++) {
    const lo = min + i * binWidth;
    const hi = lo + binWidth;
    const count = pnls.filter((p) => p >= lo && (i === bins - 1 ? p <= hi : p < hi)).length;
    const mid = (lo + hi) / 2;
    histogram.push({
      range: `$${lo.toFixed(0)}`,
      count,
      mid,
    });
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={histogram}>
        <XAxis
          dataKey="range"
          tick={{ fontSize: 9 }}
          interval={Math.max(0, Math.floor(bins / 6))}
        />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip
          formatter={(v) => [`${v} trades`, "Count"]}
          labelFormatter={(l) => l}
        />
        <ReferenceLine x={`$0`} stroke="#666" strokeDasharray="3 3" />
        <Bar dataKey="count" name="Trades">
          {histogram.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.mid >= 0 ? "#22c55e" : "#ef4444"}
              fillOpacity={0.7}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
