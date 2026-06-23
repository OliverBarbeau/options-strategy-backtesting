import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TradeLab",
  description: "Options trading backtesting, simulation, and execution",
};

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/backtest", label: "Backtest" },
  { href: "/accounts", label: "Accounts" },
  { href: "/history", label: "History" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex">
        <aside className="w-56 border-r bg-muted/40 p-4 flex flex-col gap-1 shrink-0">
          <h1 className="text-lg font-bold px-3 py-2 mb-2">TradeLab</h1>
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="px-3 py-2 rounded-md text-sm hover:bg-muted transition-colors"
            >
              {item.label}
            </Link>
          ))}
        </aside>
        <main className="flex-1 p-6 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
