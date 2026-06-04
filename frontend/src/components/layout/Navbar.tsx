"use client";
import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

type Props = {
  onOpenSidebar?: () => void;
};

export const Navbar: React.FC<Props> = ({ onOpenSidebar }) => {
  return (
    <motion.div
      className="sticky top-0 z-50 w-full border-b border-slate-200/50 bg-white/80 backdrop-blur-xl py-4 px-4 shadow-sm"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4">
        {/* Left: Logo & Branding */}
        <div className="flex items-center gap-4">
          {/* Hamburger for mobile */}
          <motion.button
            onClick={onOpenSidebar}
            className="-ml-2 mr-1 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 xl:hidden"
            aria-label="Open sidebar"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth="2" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </motion.button>

          {/* Logo */}
          <motion.div
            className="flex items-center gap-3"
            whileHover={{ scale: 1.02 }}
          >
            <div className="relative h-10 w-10 rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center text-white font-bold shadow-lg shadow-indigo-500/30">
              📦
            </div>
            <div>
              <p className="text-xs uppercase tracking-wider font-semibold text-indigo-600">Supply Chain</p>
              <div className="text-sm font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                Price Predictor
              </div>
            </div>
          </motion.div>
        </div>

        {/* Right: Navigation & CTA */}
        <div className="hidden items-center gap-6 xl:flex">
          <nav className="flex items-center gap-6">
            <Link
              href="#dashboard"
              className="text-sm font-medium text-slate-600 transition hover:text-indigo-600"
            >
              Dashboard
            </Link>
            <Link
              href="#predict"
              className="text-sm font-medium text-slate-600 transition hover:text-indigo-600"
            >
              Predict
            </Link>
          </nav>

          <motion.button
            className="ml-2 rounded-full bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:shadow-indigo-500/50"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Login
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default Navbar;
