"use client";

interface MonthlyData {
  date: string; // entry_date from trade log
  pnl: number;
}

interface MonthlyHeatmapProps {
  trades: MonthlyData[];
}

function fmt(n: number) {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

export function MonthlyHeatmap({ trades }: MonthlyHeatmapProps) {
  if (!trades || trades.length === 0) return null;

  // Aggregate P/L by month
  const monthly: Record<string, { pnl: number; trades: number; wins: number }> = {};
  for (const t of trades) {
    const month = (t.date ?? "").slice(0, 7);
    if (!month || month.length < 7) continue;
    if (!monthly[month]) monthly[month] = { pnl: 0, trades: 0, wins: 0 };
    monthly[month].pnl += t.pnl;
    monthly[month].trades += 1;
    if (t.pnl > 0) monthly[month].wins += 1;
  }

  const months = Object.keys(monthly).sort();
  if (months.length === 0) return null;

  // Get unique years
  const years = [...new Set(months.map((m) => m.slice(0, 4)))].sort();
  const monthLabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // Color scale
  const allPnl = Object.values(monthly).map((m) => m.pnl);
  const maxAbs = Math.max(1, Math.max(...allPnl.map(Math.abs)));

  function cellColor(pnl: number): string {
    const intensity = Math.min(1, Math.abs(pnl) / maxAbs);
    if (pnl >= 0) {
      const g = Math.round(100 + 155 * intensity);
      return `rgb(34, ${g}, 34)`;
    }
    const r = Math.round(100 + 155 * intensity);
    return `rgb(${r}, 34, 34)`;
  }

  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1 text-muted-foreground">Year</th>
            {monthLabels.map((m) => (
              <th key={m} className="px-2 py-1 text-muted-foreground text-center w-16">
                {m}
              </th>
            ))}
            <th className="px-2 py-1 text-muted-foreground text-right">Total</th>
          </tr>
        </thead>
        <tbody>
          {years.map((year) => {
            let yearTotal = 0;
            return (
              <tr key={year}>
                <td className="px-2 py-1 font-medium">{year}</td>
                {monthLabels.map((_, mi) => {
                  const key = `${year}-${String(mi + 1).padStart(2, "0")}`;
                  const m = monthly[key];
                  if (m) yearTotal += m.pnl;
                  return (
                    <td
                      key={key}
                      className="px-1 py-1 text-center font-mono rounded"
                      style={{
                        backgroundColor: m ? cellColor(m.pnl) : "transparent",
                        color: m ? "#fff" : "#666",
                      }}
                      title={
                        m
                          ? `${key}: ${fmt(m.pnl)} (${m.trades} trades, ${m.wins}W)`
                          : `${key}: no trades`
                      }
                    >
                      {m ? fmt(m.pnl) : "—"}
                    </td>
                  );
                })}
                <td
                  className="px-2 py-1 text-right font-mono font-bold"
                  style={{ color: yearTotal >= 0 ? "#22c55e" : "#ef4444" }}
                >
                  {fmt(yearTotal)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
