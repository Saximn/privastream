import "./globals.css";
import type { Metadata } from "next";

import Link from "next/link";

export const metadata: Metadata = {
  title: "Live Stream App",
  description: "A simple live streaming application",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white dark:bg-black">
        {/* Header */}
        <header className="border-b bg-white dark:bg-black">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold text-black dark:text-white">
                  <Link href="/" className="no-underline focus:outline-none">
                    PrivaStream
                  </Link>
                </h1>
              </div>
            </div>
          </div>
        </header>
        {children}
        {/* Footer */}
        <footer className="border-t bg-white dark:bg-black mt-16">
          <div className="container mx-auto px-4 py-8 text-center text-sm text-gray-600 dark:text-gray-400">
            <p>© 2025 Blueberry Jam. Built for TikTok TechJam 2025.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
