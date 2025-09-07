import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Live Stream App',
  description: 'A simple live streaming application',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}