"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface TickerData {
  ticker: string;
  total_pnl: number;
  trades?: number;
  win_rate?: number;
}

interface TickerBarChartProps {
  data: TickerData[];
  height?: number;
}

function fmt(n: number) {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

export function TickerBarChart({ data, height = 220 }: TickerBarChartProps) {
  if (!data || data.length === 0) return null;

  const sorted = [...data].sort((a, b) => b.total_pnl - a.total_pnl);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={sorted} layout="vertical">
        <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmt(v)} />
        <YAxis
          type="category"
          dataKey="ticker"
          tick={{ fontSize: 12, fontWeight: 600 }}
          width={50}
        />
        <Tooltip
          formatter={(v) => [fmt(Number(v)), "P/L"]}
        />
        <Bar dataKey="total_pnl" name="P/L" radius={[0, 4, 4, 0]}>
          {sorted.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.total_pnl >= 0 ? "#22c55e" : "#ef4444"}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
