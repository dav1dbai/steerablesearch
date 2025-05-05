import type { Metadata } from "next";
import "./globals.css";
import Link from 'next/link';
import { spaceGrotesk, garamond, newsreader } from './fonts';

export const metadata: Metadata = {
  title: "Weaver | Steerable Search with Claude",
  description: "Named after Warren Weaver who made Shannon's ideas accessible, Weaver helps Claude convey information in academic papers through steerable search",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.variable} ${garamond.variable} ${newsreader.variable} antialiased`}
      >
        <header className="border-b border-neutral-200 bg-white">
          <div className="max-w-screen-xl mx-auto py-5 px-4 sm:px-6 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <img src="/claude-logo.svg" alt="Claude Logo" className="h-9 w-9" />
              <Link href="/" className="text-2xl font-medium text-foreground hover:no-underline font-heading">
                Weaver
              </Link>
            </div>
            <div className="claude-pill">Steerable Search with Claude</div>
          </div>
        </header>
        <main className="max-w-screen-xl mx-auto py-8 px-4 sm:px-6">
          {children}
        </main>
        <footer className="border-t border-neutral-200 py-6 mt-14 bg-[#FFFBF8]">
          <div className="max-w-screen-xl mx-auto px-4 sm:px-6 text-center">
            <div className="flex items-center justify-center mb-4">
              <img src="/claude-logo.svg" alt="Claude Logo" className="h-6 w-6 mr-2" />
              <span className="font-heading text-lg">Weaver</span>
            </div>
            <p className="text-xs text-neutral-500 max-w-md mx-auto">
              Helping Claude convey research through steerable search,<br />inspired by Weaver's role in making information theory accessible
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
