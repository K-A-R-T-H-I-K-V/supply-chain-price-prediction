"use client";

import React, { useCallback, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import OnboardingChat from '@/components/OnboardingChat';
import PredictForm from '@/components/PredictForm';

type View = 'landing' | 'prediction';

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

const ModuleCard = ({
  title,
  description,
  icon,
  badge,
  enabled,
  onOpen,
}: {
  title: string;
  description: string;
  icon: string;
  badge?: string;
  enabled: boolean;
  onOpen: () => void;
}) => (
  <motion.button
    type="button"
    variants={fadeUp}
    disabled={!enabled}
    onClick={onOpen}
    className={`group relative flex w-full flex-col rounded-2xl border p-6 text-left transition-all duration-300 ${
      enabled
        ? 'border-white/60 bg-white/60 shadow-xl shadow-indigo-200/30 backdrop-blur-md hover:border-indigo-300/60 hover:bg-white/80 hover:shadow-indigo-300/40'
        : 'cursor-not-allowed border-slate-200/40 bg-white/30 backdrop-blur-sm opacity-60'
    }`}
    whileHover={enabled ? { y: -5, scale: 1.01 } : undefined}
    whileTap={enabled   ? { scale: 0.98 }         : undefined}
  >
    {badge && (
      <span className="absolute right-4 top-4 rounded-full border border-slate-200/60 bg-white/70 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 backdrop-blur-sm">
        {badge}
      </span>
    )}
    <div
      className={`flex h-12 w-12 items-center justify-center rounded-xl text-xl text-white shadow-lg transition-transform duration-300 ${
        enabled
          ? 'bg-gradient-to-br from-indigo-500 to-violet-600 shadow-indigo-500/30 group-hover:scale-110'
          : 'bg-gradient-to-br from-slate-300 to-slate-400 shadow-slate-300/30'
      }`}
    >
      {icon}
    </div>

    <h3 className="mt-4 text-base font-bold tracking-tight text-slate-900">{title}</h3>
    <p className="mt-1.5 text-sm leading-6 text-slate-500">{description}</p>

    <span
      className={`mt-5 inline-flex items-center gap-1 text-sm font-semibold transition-colors ${
        enabled
          ? 'text-indigo-600 group-hover:text-indigo-700'
          : 'text-slate-400'
      }`}
    >
      {enabled ? (
        <>Open module <motion.span animate={{ x: [0, 4, 0] }} transition={{ repeat: Infinity, duration: 1.4 }}>→</motion.span></>
      ) : (
        'Complete intro first'
      )}
    </span>
  </motion.button>
);

const PredictionModuleBanner = ({ onBack }: { onBack: () => void }) => (
  <motion.div
    className="relative mb-6 overflow-hidden rounded-2xl border border-white/20 bg-gradient-to-r from-indigo-600 via-violet-600 to-indigo-700 px-6 py-8 text-white shadow-2xl shadow-indigo-600/30 sm:px-8"
    initial={{ opacity: 0, y: -16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
  >
    {/* decorative blobs */}
    <div className="pointer-events-none absolute -right-12 -top-12 h-48 w-48 rounded-full bg-white/10 blur-3xl" />
    <div className="pointer-events-none absolute -bottom-8 left-24 h-32 w-32 rounded-full bg-violet-400/20 blur-2xl" />

    <div className="relative flex flex-wrap items-start justify-between gap-4">
      <div>
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-200">Active module</p>
        <h1 className="mt-2 text-2xl font-bold sm:text-3xl">Price Prediction Module</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-indigo-100/90">
          Forecast unit prices from order parameters and get AI-powered expert suggestions with live market context.
        </p>
      </div>
      <button
        type="button"
        onClick={onBack}
        className="shrink-0 rounded-full border border-white/25 bg-white/10 px-4 py-2 text-sm font-semibold text-white backdrop-blur-sm transition hover:bg-white/20"
      >
        ← Back to home
      </button>
    </div>
  </motion.div>
);

export const PredictionWorkspace: React.FC = () => {
  const [view, setView] = useState<View>('landing');
  const [introComplete, setIntroComplete] = useState(false);
  const [introKey, setIntroKey] = useState(0);

  const openPredictionModule = useCallback(() => {
    setView('prediction');
    requestAnimationFrame(() => {
      document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' });
    });
  }, []);

  const goToLanding = useCallback(() => {
    setView('landing');
    setIntroComplete(false);
    setIntroKey((k) => k + 1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  useEffect(() => {
    const onOpen  = () => { if (introComplete) openPredictionModule(); else goToLanding(); };
    const onReplay = () => goToLanding();

    window.addEventListener('open-prediction-module', onOpen);
    window.addEventListener('replay-intro', onReplay);
    window.addEventListener('go-to-landing', onReplay);
    return () => {
      window.removeEventListener('open-prediction-module', onOpen);
      window.removeEventListener('replay-intro', onReplay);
      window.removeEventListener('go-to-landing', onReplay);
    };
  }, [goToLanding, introComplete, openPredictionModule]);

  return (
    <div id="dashboard" className="w-full">
      <AnimatePresence mode="wait">

        {view === 'landing' ? (
          <motion.div
            key="landing"
            variants={stagger}
            initial="hidden"
            animate="show"
            exit={{ opacity: 0, y: -16, transition: { duration: 0.25 } }}
            className="mx-auto max-w-4xl space-y-10"
          >
            {/* Hero heading */}
            <motion.div variants={fadeUp} className="text-center">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-200/60 bg-indigo-50/80 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-indigo-600 backdrop-blur-sm">
                <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                GenAI · Supply Chain
              </span>
              <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl lg:text-5xl">
                Supply Chain{' '}
                <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
                  Control Center
                </span>
              </h1>
              <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-slate-500">
                Complete the short intro below, then open the prediction module to start forecasting unit prices with live market context.
              </p>
            </motion.div>

            {/* Intro card — compact with glass */}
            <motion.section variants={fadeUp}>
              <p className="mb-2.5 text-[11px] font-bold uppercase tracking-[0.15em] text-slate-400">Step 1 · Quick intro</p>
              <div className="rounded-2xl border border-white/60 bg-white/50 shadow-xl shadow-slate-200/40 backdrop-blur-md">
                <OnboardingChat key={introKey} onComplete={() => setIntroComplete(true)} />
              </div>
            </motion.section>

            {/* Module cards */}
            <motion.section variants={stagger}>
              <motion.p variants={fadeUp} className="mb-3 text-[11px] font-bold uppercase tracking-[0.15em] text-slate-400">
                Step 2 · Choose a module
              </motion.p>
              <div className="grid gap-4 sm:grid-cols-2">
                <ModuleCard
                  icon="⚡"
                  title="Price Prediction"
                  description="Enter order details to get a unit price forecast and AI-powered expert suggestions based on live data."
                  enabled={introComplete}
                  onOpen={openPredictionModule}
                />
                <ModuleCard
                  icon="📊"
                  title="Market Insights"
                  description="Historical trends, category benchmarks, and weather-informed analytics — coming in the next release."
                  badge="Soon"
                  enabled={false}
                  onOpen={() => undefined}
                />
              </div>
            </motion.section>
          </motion.div>

        ) : (
          <motion.div
            key="prediction"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16, transition: { duration: 0.25 } }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            <PredictionModuleBanner onBack={goToLanding} />
            <PredictForm />
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
};

export default PredictionWorkspace;
