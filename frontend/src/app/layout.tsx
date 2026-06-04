import type { Metadata } from 'next';
import DashboardWrapper from '@/components/layout/DashboardWrapper';
import './globals.css';

export const metadata: Metadata = {
  title: 'Supply Chain Price Prediction',
  description: 'Dashboard for supply chain price prediction',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">
        <DashboardWrapper>{children}</DashboardWrapper>
      </body>
    </html>
  );
}
