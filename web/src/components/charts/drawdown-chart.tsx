"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface DrawdownPoint {
  date: string;
  equity: number;
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
  height?: number;
}

export function DrawdownChart({ data, height = 200 }: DrawdownChartProps) {
  if (!data || data.length < 2) return null;

  // Compute drawdown series from equity curve
  const ddData = (() => {
    let peak = data[0].equity;
    return data.map((d) => {
      peak = Math.max(peak, d.equity);
      const dd = ((d.equity - peak) / peak) * 100;
      return { date: d.date.slice(0, 10), dd, equity: d.equity };
    });
  })();

  const minDD = Math.min(...ddData.map((d) => d.dd));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={ddData}>
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10 }}
          tickFormatter={(v) => v.slice(5)}
        />
        <YAxis
          tick={{ fontSize: 10 }}
          tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          domain={[Math.floor(minDD / 5) * 5, 0]}
        />
        <Tooltip
          formatter={(v) => `${Number(v).toFixed(1)}%`}
          labelFormatter={(l) => l}
        />
        <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
        <ReferenceLine
          y={-20}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label={{ value: "-20%", fill: "#ef4444", fontSize: 10 }}
        />
        <Area
          type="monotone"
          dataKey="dd"
          stroke="#ef4444"
          fill="#ef4444"
          fillOpacity={0.15}
          name="Drawdown"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
