"use client";
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predict, getExpertSuggestions } from '@/services/api';
import type { PredictRequest } from '@/types';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import OnboardingChat from '@/components/OnboardingChat';

const ONBOARDING_KEY = 'scpp_onboarding_complete';

// Skeleton component for prediction result
const PredictionSkeleton = () => (
  <motion.div
    className="mt-10 rounded-2xl border border-indigo-200/50 bg-gradient-to-br from-indigo-50/80 to-violet-50/80 backdrop-blur-sm px-8 py-8 shadow-lg shadow-indigo-200/20"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4 }}
  >
    <p className="text-xs uppercase tracking-widest font-semibold text-indigo-700">Prediction Result</p>
    <div className="mt-4 h-16 w-48 animate-pulse rounded-lg bg-gradient-to-r from-indigo-200 via-indigo-100 to-indigo-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
    <div className="mt-3 h-12 w-full max-w-xs animate-pulse rounded-lg bg-gradient-to-r from-indigo-200 via-indigo-100 to-indigo-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
  </motion.div>
);

// Skeleton component for suggestions cards
const SuggestionCardSkeleton = () => (
  <motion.div
    className="rounded-2xl border border-slate-200/50 bg-white/50 backdrop-blur-sm shadow-lg shadow-slate-200/10"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
  >
    <div className="px-6 py-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 space-y-3">
          <div className="h-4 w-40 animate-pulse rounded-lg bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
          <div className="flex flex-wrap gap-2">
            <div className="h-6 w-16 animate-pulse rounded-full bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
            <div className="h-6 w-20 animate-pulse rounded-full bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
          </div>
        </div>
        <div className="h-6 w-6 animate-pulse rounded bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%]" style={{ animation: 'shimmer 2s infinite' }} />
      </div>
    </div>
  </motion.div>
);

const SuggestionsSkeleton = () => (
  <div className="space-y-4">
    {Array.from({ length: 3 }).map((_, i) => (
      <motion.div
        key={i}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: i * 0.1 }}
      >
        <SuggestionCardSkeleton />
      </motion.div>
    ))}
  </div>
);

// Suggestion Card Component
interface SuggestionItem {
  raw: string;
  title: string;
  detail: string;
  sourceTags: string[];
}

interface SuggestionCardProps {
  item: SuggestionItem;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  getTagColorClasses: (tag: string) => string;
}

const SuggestionCard: React.FC<SuggestionCardProps> = ({
  item,
  index,
  isExpanded,
  onToggle,
  getTagColorClasses,
}) => (
  <motion.div
    className="rounded-2xl border border-slate-200/50 bg-white/70 backdrop-blur-sm shadow-lg shadow-slate-200/10 overflow-hidden transition hover:shadow-lg hover:shadow-slate-300/20"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.08 }}
    whileHover={{ y: -2 }}
  >
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={isExpanded}
      className="w-full text-left px-6 py-5"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <p className="text-sm font-bold text-slate-900">{item.title}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {item.sourceTags.map((tag) => (
              <motion.span
                key={tag}
                className={`rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${getTagColorClasses(tag)}`}
                whileHover={{ scale: 1.08 }}
                transition={{ type: 'spring', stiffness: 400, damping: 10 }}
              >
                {tag}
              </motion.span>
            ))}
          </div>
        </div>
        <motion.span
          className="text-lg text-indigo-500 font-bold"
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          ⌄
        </motion.span>
      </div>
    </button>

    <AnimatePresence>
      {isExpanded && item.detail && (
        <motion.div
          className="border-t border-slate-200/50 bg-gradient-to-br from-slate-50/50 to-indigo-50/30 px-6 py-5"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
        >
          <p className="text-sm leading-6 text-slate-700">{item.detail}</p>
        </motion.div>
      )}
    </AnimatePresence>
  </motion.div>
);

const categories = ['Office Supplies', 'Technology', 'Furniture'] as const;
const shipModes = ['Regular Air', 'Delivery Truck', 'Express Air'] as const;
const orderPriorities = ['Low', 'Medium', 'High', 'Critical'] as const;
const months = [
  { value: 1, label: 'Jan' },
  { value: 2, label: 'Feb' },
  { value: 3, label: 'Mar' },
  { value: 4, label: 'Apr' },
  { value: 5, label: 'May' },
  { value: 6, label: 'Jun' },
  { value: 7, label: 'Jul' },
  { value: 8, label: 'Aug' },
  { value: 9, label: 'Sep' },
  { value: 10, label: 'Oct' },
  { value: 11, label: 'Nov' },
  { value: 12, label: 'Dec' },
] as const;

type PredictFormState = {
  order_quantity: string;
  discount: string;
  shipping_cost: string;
  product_base_margin: string;
  product_category: (typeof categories)[number];
  month: number;
  ship_mode: (typeof shipModes)[number];
  order_priority: (typeof orderPriorities)[number];
};

export const PredictForm: React.FC = () => {
  const [onboardingComplete, setOnboardingComplete] = useState(false);
  const [introKey, setIntroKey] = useState(0);
  const [form, setForm] = useState<PredictFormState>({
    order_quantity: '',
    discount: '',
    shipping_cost: '',
    product_base_margin: '',
    product_category: 'Office Supplies',
    month: 1,
    ship_mode: 'Regular Air',
    order_priority: 'Medium',
  });
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expertOpen, setExpertOpen] = useState(false);
  const [expertLoading, setExpertLoading] = useState(false);
  const [expertResult, setExpertResult] = useState<{ message: string; suggestions: string[]; summary?: string } | null>(null);
  const [expertError, setExpertError] = useState<string | null>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && localStorage.getItem(ONBOARDING_KEY) === 'true') {
      setOnboardingComplete(true);
    }
  }, []);

  const handleOnboardingComplete = useCallback(() => {
    setOnboardingComplete(true);
  }, []);

  const replayIntro = useCallback(() => {
    localStorage.removeItem(ONBOARDING_KEY);
    setOnboardingComplete(false);
    setIntroKey((k) => k + 1);
  }, []);

  const handleChange = <K extends keyof PredictFormState>(key: K, value: PredictFormState[K]) => {
    setForm((s) => ({ ...s, [key]: value }));
  };

  const resetForm = () => {
    setForm({
      order_quantity: '',
      discount: '',
      shipping_cost: '',
      product_base_margin: '',
      product_category: 'Office Supplies',
      month: 1,
      ship_mode: 'Regular Air',
      order_priority: 'Medium',
    });
    setPrediction(null);
    setError(null);
    setExpertOpen(false);
    setExpertError(null);
    setExpertResult(null);
  };

  useEffect(() => {
    const resetHandler = () => resetForm();
    const restartHandler = () => {
      localStorage.removeItem(ONBOARDING_KEY);
      setOnboardingComplete(false);
      setIntroKey((k) => k + 1);
      resetForm();
    };
    const replayHandler = () => replayIntro();
    window.addEventListener('reset-prediction-form', resetHandler);
    window.addEventListener('restart-onboarding', restartHandler);
    window.addEventListener('replay-intro', replayHandler);
    return () => {
      window.removeEventListener('reset-prediction-form', resetHandler);
      window.removeEventListener('restart-onboarding', restartHandler);
      window.removeEventListener('replay-intro', replayHandler);
    };
  }, [replayIntro]);

  const buildPayload = (): PredictRequest => ({
    order_quantity: Number(form.order_quantity),
    discount: Number(form.discount),
    shipping_cost: Number(form.shipping_cost),
    product_base_margin: Number(form.product_base_margin),
    product_category: form.product_category,
    month: form.month,
    ship_mode: form.ship_mode,
    order_priority: form.order_priority,
  });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    setExpertError(null);
    setExpertResult(null);
    setExpertOpen(false);

    const payload = buildPayload();

    if (
      Number.isNaN(payload.order_quantity) ||
      Number.isNaN(payload.discount) ||
      Number.isNaN(payload.shipping_cost) ||
      Number.isNaN(payload.product_base_margin) ||
      payload.order_quantity < 1
    ) {
      setError('Please enter valid numeric values for quantity, discount, shipping cost, and margin.');
      setLoading(false);
      return;
    }

    try {
      const res = await predict(payload);
      setPrediction(res.predicted_unit_price);
      window.dispatchEvent(new Event('history-updated'));
    } catch (err: any) {
      console.error('predict failed', err);
      setError(err?.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const loadExpertSuggestions = async () => {
    if (prediction === null) {
      setExpertError('Generate a prediction first to get expert suggestions.');
      return;
    }

    setExpertLoading(true);
    setExpertError(null);
    setExpandedIndex(null);

    try {
      const payload = {
        ...buildPayload(),
        predicted_price: prediction,
      };
      const res = await getExpertSuggestions(payload);
      setExpertResult(res);
      setExpertOpen(true);
    } catch (err: any) {
      console.error('expert suggestions failed', err);
      setExpertError(err?.message || 'Unable to load expert suggestions.');
    } finally {
      setExpertLoading(false);
    }
  };

  const getTagColorClasses = (tag: string): string => {
    const normalizedTag = tag.toLowerCase().trim();
    if (normalizedTag === 'news') return 'border-red-300/50 bg-red-100/60 text-red-700 font-semibold';
    if (normalizedTag === 'weather') return 'border-blue-300/50 bg-blue-100/60 text-blue-700 font-semibold';
    if (normalizedTag === 'finance') return 'border-green-300/50 bg-green-100/60 text-green-700 font-semibold';
    if (normalizedTag === 'internal') return 'border-slate-300/50 bg-slate-100/60 text-slate-700 font-semibold';
    return 'border-slate-300/50 bg-slate-100/60 text-slate-700 font-semibold';
  };

  useEffect(() => {
    const handleLoadPrediction = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.form && detail?.predicted_price) {
        setForm({
          order_quantity: String(detail.form.order_quantity),
          discount: String(detail.form.discount),
          shipping_cost: String(detail.form.shipping_cost),
          product_base_margin: String(detail.form.product_base_margin),
          product_category: detail.form.product_category,
          month: detail.form.month,
          ship_mode: detail.form.ship_mode,
          order_priority: detail.form.order_priority,
        });
        setPrediction(detail.predicted_price);
        setExpertOpen(false);
        setExpertResult(null);
      }
    };
    window.addEventListener('load-prediction', handleLoadPrediction);
    return () => window.removeEventListener('load-prediction', handleLoadPrediction);
  }, []);

  const suggestionItems = useMemo(() => {
    if (!expertResult) return [];

    return expertResult.suggestions.slice(0, 5).map((raw) => {
      const tagMatch = raw.match(/\[([^\]]+)\]$/);
      const sourceTags = tagMatch ? tagMatch[1].split(',').map((tag) => tag.trim()) : [];
      let text = raw.replace(/\[([^\]]+)\]$/, '').trim();
      text = text.replace(/^Suggestion\s*\d+\s*:\s*/i, '').trim();

      let title = '';
      let detail = text;
      const boldTitleMatch = text.match(/^\*\*(.+?)\*\*\s*[:\-–—]?\s*/);
      if (boldTitleMatch) {
        title = boldTitleMatch[1].trim();
        detail = text.slice(boldTitleMatch[0].length).trim();
      } else {
        const sentenceEnd = text.search(/\.|\n/);
        if (sentenceEnd > 0 && sentenceEnd < 120) {
          title = text.slice(0, sentenceEnd + 1).trim();
          detail = text.slice(sentenceEnd + 1).trim();
        } else {
          title = text;
          detail = '';
        }
      }

      if (!detail) {
        detail = title;
      }

      return {
        raw,
        title,
        detail,
        sourceTags,
      };
    });
  }, [expertResult]);

  return (
    <div id="predict" className="grid gap-6 lg:grid-cols-[1fr_360px] h-full">
      {/* LEFT PANEL: Control Center */}
      <motion.div
        className="w-full"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        {!onboardingComplete ? (
          <OnboardingChat key={introKey} onComplete={handleOnboardingComplete} />
        ) : (
        <Card className="w-full rounded-2xl border border-slate-200/50 bg-white shadow-xl shadow-slate-200/30 backdrop-blur-sm p-0 overflow-hidden">
          {/* Header Section */}
          <div className="border-b border-slate-200/50 bg-gradient-to-r from-slate-50/80 to-indigo-50/80 backdrop-blur-sm px-8 py-8">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-widest font-bold text-indigo-600">Control Center</p>
                  <h2 className="mt-2 text-3xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                    Quick Predict
                  </h2>
                  <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
                    Input your supply chain parameters and generate a price prediction. Expert suggestions appear in the right panel.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={replayIntro}
                  className="shrink-0 rounded-full border border-indigo-200 bg-white px-4 py-2 text-xs font-semibold text-indigo-600 transition hover:bg-indigo-50"
                >
                  See intro again
                </button>
              </div>
            </motion.div>
          </div>

          {/* Form Section */}
          <form onSubmit={submit} className="p-8">
            {/* Input Grid - Strict 2-Column Layout */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
              {/* Order Quantity */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Order Quantity</label>
                <input
                  type="number"
                  min={1}
                  value={form.order_quantity}
                  placeholder="e.g. 12"
                  onChange={(e) => handleChange('order_quantity', e.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder-slate-400 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                />
              </motion.div>

              {/* Discount */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Discount Rate</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.discount}
                  placeholder="e.g. 0.05"
                  onChange={(e) => handleChange('discount', e.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder-slate-400 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                />
              </motion.div>

              {/* Shipping Cost */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Shipping Cost</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.shipping_cost}
                  placeholder="e.g. 35"
                  onChange={(e) => handleChange('shipping_cost', e.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder-slate-400 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                />
              </motion.div>

              {/* Product Margin */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Product Margin</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.product_base_margin}
                  placeholder="e.g. 0.50"
                  onChange={(e) => handleChange('product_base_margin', e.target.value)}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder-slate-400 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                />
              </motion.div>

              {/* Category */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Category</label>
                <select
                  value={form.product_category}
                  onChange={(e) => handleChange('product_category', e.target.value as PredictFormState['product_category'])}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                >
                  {categories.map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </motion.div>

              {/* Month */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.35 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Month</label>
                <select
                  value={form.month}
                  onChange={(e) => handleChange('month', Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                >
                  {months.map((month) => (
                    <option key={month.value} value={month.value}>
                      {month.label}
                    </option>
                  ))}
                </select>
              </motion.div>

              {/* Ship Mode */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Shipping Mode</label>
                <select
                  value={form.ship_mode}
                  onChange={(e) => handleChange('ship_mode', e.target.value as PredictFormState['ship_mode'])}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                >
                  {shipModes.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </motion.div>

              {/* Order Priority */}
              <motion.div
                className="flex flex-col"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 }}
              >
                <label className="mb-3 block text-sm font-semibold text-slate-700">Order Priority</label>
                <select
                  value={form.order_priority}
                  onChange={(e) => handleChange('order_priority', e.target.value as PredictFormState['order_priority'])}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                >
                  {orderPriorities.map((priority) => (
                    <option key={priority} value={priority}>
                      {priority}
                    </option>
                  ))}
                </select>
              </motion.div>
            </div>

            {/* Button Group */}
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between pt-2 border-t border-slate-200/50">
              <motion.div
                className="pt-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center rounded-xl px-6 py-3 text-sm font-bold text-white bg-gradient-to-r from-indigo-600 to-violet-600 shadow-lg shadow-indigo-500/30 transition hover:shadow-indigo-500/50 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <motion.span
                        className="inline-block mr-2"
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      >
                        ⚡
                      </motion.span>
                      Predicting…
                    </>
                  ) : (
                    '⚡ Generate Prediction'
                  )}
                </button>
              </motion.div>

              {prediction !== null && (
                <motion.button
                  type="button"
                  onClick={loadExpertSuggestions}
                  disabled={expertLoading}
                  className="inline-flex items-center justify-center rounded-xl border-2 border-indigo-200 bg-white px-6 py-3 text-sm font-bold text-indigo-600 transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-60"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {expertLoading ? (
                    <>
                      <motion.span
                        className="inline-block mr-2"
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      >
                        🤖
                      </motion.span>
                      Expert Advice…
                    </>
                  ) : (
                    '🤖 Get Expert Suggestions'
                  )}
                </motion.button>
              )}
            </div>

            {/* Error Message */}
            <AnimatePresence>
              {error && (
                <motion.div
                  className="mt-6 p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </form>

          {/* Prediction Result */}
          <AnimatePresence>
            {loading && (
              <motion.div className="px-8 pb-8">
                <PredictionSkeleton />
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {!loading && prediction !== null && (
              <motion.div
                className="px-8 pb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.4 }}
              >
                <div className="rounded-2xl border border-indigo-200/50 bg-gradient-to-br from-indigo-50/80 via-violet-50/50 to-indigo-50/80 backdrop-blur-sm px-8 py-8 shadow-lg shadow-indigo-200/20">
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
                    <p className="text-xs uppercase tracking-widest font-bold text-indigo-700">Predicted Unit Price</p>
                    <motion.p
                      className="mt-4 text-6xl font-bold text-slate-900"
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 200 }}
                    >
                      ${prediction.toFixed(2)}
                    </motion.p>
                    <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-700">
                      This prediction is based on your input parameters. Review expert suggestions in the right panel for actionable insights.
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
        )}
      </motion.div>

      {/* RIGHT PANEL: Expressive Zone with Glassmorphism */}
      <motion.section
        className="flex flex-col gap-4 h-fit sticky top-24 max-h-[calc(100vh-140px)] overflow-y-auto scrollbar-thin scrollbar-thumb-indigo-300 scrollbar-track-slate-100"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        {/* Suggestions Panel Header */}
        <motion.div
          className="rounded-2xl border border-slate-200/50 bg-white/80 backdrop-blur-xl p-6 shadow-lg shadow-slate-200/20"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <p className="text-xs uppercase tracking-widest font-bold text-slate-600">Insights Zone</p>
          <h3 className="mt-2 text-xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
            Expert Suggestions
          </h3>
        </motion.div>

        {/* Suggestions Content */}
        <motion.div className="space-y-4 flex-1 overflow-y-auto pr-2">
          {expertLoading && <SuggestionsSkeleton />}

          <AnimatePresence>
            {expertError && (
              <motion.div
                className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                {expertError}
              </motion.div>
            )}
          </AnimatePresence>

          {!onboardingComplete && (
            <motion.div
              className="rounded-2xl border border-indigo-200/50 bg-gradient-to-br from-indigo-50/80 to-violet-50/50 backdrop-blur-sm p-6 text-sm text-slate-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <p className="font-semibold text-slate-900">What happens next?</p>
              <p className="mt-2 leading-6">
                Complete the short intro on the left, then fill in your order details to get a price forecast and AI expert suggestions here.
              </p>
            </motion.div>
          )}

          {onboardingComplete && !expertOpen && prediction === null && !expertLoading && (
            <motion.div
              className="rounded-2xl border border-slate-200/50 bg-gradient-to-br from-slate-50/80 to-slate-50/50 backdrop-blur-sm p-4 text-sm text-slate-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Generate a prediction first to see expert suggestions.
            </motion.div>
          )}

          {!expertOpen && prediction !== null && !expertLoading && !expertResult && (
            <motion.div
              className="rounded-2xl border border-slate-200/50 bg-gradient-to-br from-indigo-50/80 to-indigo-50/50 backdrop-blur-sm p-4 text-sm text-slate-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Click "Get Expert Suggestions" to populate this panel with AI-driven insights.
            </motion.div>
          )}

          {!expertLoading && expertResult && (
            <AnimatePresence mode="wait">
              {expertResult.message && (
                <motion.div
                  key="expert-message"
                  className="rounded-2xl border border-slate-200/50 bg-white/60 backdrop-blur-sm px-6 py-5 text-sm leading-6 text-slate-700 shadow-lg shadow-slate-200/10"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {expertResult.message}
                </motion.div>
              )}

              <motion.div
                key="suggestions-list"
                className="space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {suggestionItems.map((item, index) => (
                  <SuggestionCard
                    key={`suggestion-${index}-${item.title}`}
                    item={item}
                    index={index}
                    isExpanded={expandedIndex === index}
                    onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
                    getTagColorClasses={getTagColorClasses}
                  />
                ))}
              </motion.div>

              {expertResult.summary && (
                <motion.div
                  key="expert-summary"
                  className="mt-4 rounded-2xl border border-slate-200/50 bg-gradient-to-br from-slate-50/80 to-slate-50/50 backdrop-blur-sm px-6 py-5 text-sm text-slate-700 shadow-lg shadow-slate-200/10"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <p className="font-bold text-slate-900">Quick Summary</p>
                  <p className="mt-2">{expertResult.summary}</p>
                </motion.div>
              )}
            </AnimatePresence>
          )}
        </motion.div>
      </motion.section>
    </div>
  );
};

export default PredictForm;
