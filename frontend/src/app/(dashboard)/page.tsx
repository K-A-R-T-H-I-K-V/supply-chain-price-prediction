"use client";
import { motion } from 'framer-motion';
import PredictForm from '@/components/PredictForm';

export default function DashboardPage() {
  return (
    <motion.main
      className="min-h-[calc(100vh-88px)] flex items-stretch justify-center px-4 py-8 sm:px-6 lg:px-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="w-full max-w-[1600px]">
        <PredictForm />
      </div>
    </motion.main>
  );
}

