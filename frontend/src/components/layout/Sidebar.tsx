"use client";
import Link from 'next/link';
import React, { useEffect, useState } from 'react';
import { getHistory } from '@/services/api';
import supabase from '@/lib/supabaseClient';
import Auth from '@/components/Auth';
import api from '@/services/api';
import type { PredictionHistoryItem } from '@/types';

type Props = {
  onClose?: () => void;
  className?: string;
};

const nav = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'predict', label: 'Predict' },
  { id: 'weather', label: 'Weather' },
  { id: 'insights', label: 'Insights' },
  { id: 'history', label: 'History' },
];

export const Sidebar: React.FC<Props> = ({ onClose, className }) => {
  const wrapperClass = className ?? 'hidden xl:block w-72 shrink-0';
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);

  const fetchHistory = async () => {
    try {
      const items = await getHistory();
      setHistory(items.reverse());
    } catch (error) {
      console.error('Unable to load prediction history', error);
    }
  };

  useEffect(() => {
    fetchHistory();
    const handler = () => fetchHistory();
    window.addEventListener('history-updated', handler);
    return () => window.removeEventListener('history-updated', handler);
  }, []);

  useEffect(() => {
    // If user is signed in, attach token to API client for backend requests
    const setAuthHeader = async () => {
      try {
        const { data } = await supabase.auth.getSession();
        const token = data?.session?.access_token;
        if (token) {
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        }
      } catch (e) {
        // ignore
      }
    };
    setAuthHeader();
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.access_token) api.defaults.headers.common['Authorization'] = `Bearer ${session.access_token}`;
      else delete api.defaults.headers.common['Authorization'];
    });
    const subscription = (data as any)?.subscription;
    return () => subscription?.unsubscribe?.();
  }, []);

  const handleNewPrediction = () => {
    window.dispatchEvent(new Event('reset-prediction-form'));
    document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleHistoryClick = (item: PredictionHistoryItem) => {
    window.dispatchEvent(
      new CustomEvent('load-prediction', {
        detail: {
          form: item.request,
          predicted_price: item.predicted_unit_price,
        },
      })
    );
    document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <aside className={wrapperClass}>
      <div className="flex h-full flex-col gap-6 rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <div>
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-3xl bg-indigo-600 text-2xl text-white shadow-lg">
              🚚
            </div>
            <h2 className="text-2xl font-semibold text-slate-900">Supply Chain</h2>
            <p className="mt-2 text-sm text-slate-500">Price prediction dashboard</p>
          </div>
          {onClose && (
            <button onClick={onClose} aria-label="Close sidebar" className="ml-4 rounded-md bg-slate-50 px-2 py-1 text-sm text-slate-600">
              ✕
            </button>
          )}
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <button
            type="button"
            onClick={handleNewPrediction}
            className="flex items-center justify-center gap-2 rounded-3xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-700"
          >
            <span className="text-lg">+</span>
            New prediction
            </button>
            <div>
              <Auth />
            </div>
          </div>

          <Link
            href="#history"
            className="flex items-center gap-3 rounded-3xl border border-transparent bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-indigo-50"
          >
            History
          </Link>
        </div>

        <div id="history" className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <p className="font-semibold text-slate-900">Prediction history</p>
          <p className="mt-3 text-sm text-slate-600">
            Past forecasts are logged here automatically. Click New prediction to reset the form and run another scenario.
          </p>

          {history.length === 0 ? (
            <div className="mt-4 rounded-3xl border border-slate-200 bg-white px-4 py-4 text-sm text-slate-700 shadow-sm">
              No history yet. Run a prediction and expert suggestions will be stored here.
            </div>
          ) : (
            <ul className="mt-4 space-y-3">
              {history.slice(0, 4).map((item, index) => (
                <li
                  key={`${item.timestamp}-${index}`}
                  onClick={() => handleHistoryClick(item)}
                  className="cursor-pointer rounded-3xl border border-slate-200 bg-white px-4 py-4 text-sm text-slate-700 shadow-sm transition hover:border-indigo-300 hover:bg-indigo-50 hover:shadow-md"
                >
                  <p className="font-semibold text-slate-900">${item.predicted_unit_price.toFixed(2)}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.24em] text-slate-500">
                    {new Date(item.timestamp).toLocaleString()}
                  </p>
                  <p className="mt-3 text-sm text-slate-600">
                    {item.request.product_category}, {item.request.order_priority} priority
                  </p>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="mt-auto rounded-[1.25rem] border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <p className="font-semibold text-slate-900">Mini Storage</p>
          <p className="mt-1 text-sm">Click any history item to restore the prediction and form data.</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
