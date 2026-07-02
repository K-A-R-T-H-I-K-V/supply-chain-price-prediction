"use client";
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import Navbar from './Navbar';

export const DashboardWrapper: React.FC<React.PropsWithChildren<Record<string, unknown>>> = ({ children }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="mesh-background relative min-h-screen text-slate-900 overflow-hidden">
      <Navbar onOpenSidebar={() => setOpen(true)} />

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="fixed inset-0 z-40 flex"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="fixed inset-0 bg-black/40 backdrop-blur-sm"
              onClick={() => setOpen(false)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
            <motion.div
              className="relative h-full w-72 p-4 overflow-hidden"
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            >
              <Sidebar onClose={() => setOpen(false)} className="!block h-full" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-6 px-4 py-8 sm:px-6 lg:px-8 xl:grid-cols-[260px_1fr] min-h-[calc(100vh-88px)]">
        <motion.div
          className="hidden xl:block"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Sidebar />
        </motion.div>
        <motion.main
          className="min-h-[70vh]"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          {children}
        </motion.main>
      </div>
    </div>
  );
};

export default DashboardWrapper;
