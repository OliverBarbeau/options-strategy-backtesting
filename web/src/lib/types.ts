export interface AccountSummary {
  name: string;
  strategy: string;
  account_type: "live" | "backtest";
  starting_capital: number;
  equity: number;
  balance: number;
  locked: number;
  total_pnl: number;
  total_return_pct: number;
  total_trades: number;
  win_rate: number;
  open_positions: number;
  last_advanced_date: string;
}

export interface PositionDetail {
  id: string;
  ticker: string;
  entry_date: string;
  entry_price: number;
  short_strike: number;
  long_strike: number;
  contracts: number;
  collateral: number;
  credit_received: number;
  buffer: number;
  close_target_date: string;
  notes: string;
}

export interface MtmPosition {
  id: string;
  ticker: string;
  entry_date: string;
  entry_price: number;
  current_price: number;
  price_change: number;
  short_strike: number;
  buffer_remaining: number;
  contracts: number;
  credit: number;
  close_cost: number;
  unrealized_pnl: number;
  dte_remaining: number;
  close_target: string;
}

export interface TradeDetail {
  id: string;
  ticker: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  contracts: number;
  pnl: number;
  winner: boolean;
  exit_reason: string;
  buffer: number;
}

export interface AccountDetail {
  summary: AccountSummary;
  positions: PositionDetail[];
  recent_trades: TradeDetail[];
  equity_curve: { date: string; equity: number; positions: number }[];
  config_summary: string;
}

export interface BacktestMetrics {
  total_trades: number;
  winners: number;
  losers: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  max_drawdown_pct: number;
  best_trade: number;
  worst_trade: number;
  avg_winner: number;
  avg_loser: number;
}

export interface TickerResult {
  ticker: string;
  trades: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
}

export interface TradeEntry {
  date: string;
  exit_date: string;
  ticker: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  winner: boolean;
  contracts: number;
  credit: number;
  sigma: number;
  buffer_used?: number;
  pullback_pct?: number;
  stacked?: boolean;
}

export interface BacktestResponse {
  id: string;
  status: string;
  strategy: string;
  tickers: string[];
  parameters: Record<string, number>;
  date_range: { start: string; end: string };
  metrics: BacktestMetrics | null;
  ticker_results: TickerResult[];
  trade_log: TradeEntry[];
  equity_curve: { date: string; equity: number }[];
  duration_seconds: number;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  defaults: Record<string, number>;
}
