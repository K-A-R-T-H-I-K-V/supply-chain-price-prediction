"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import supabase from '@/lib/supabaseClient';

type AuthMode = 'login' | 'signup';

const GoogleIcon = () => (
  <svg className="h-5 w-5" viewBox="0 0 24 24" aria-hidden>
    <path
      fill="#4285F4"
      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
    />
    <path
      fill="#34A853"
      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
    />
    <path
      fill="#FBBC05"
      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
    />
    <path
      fill="#EA4335"
      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
    />
  </svg>
);

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<AuthMode>('signup');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [fullName, setFullName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    const checkSession = async () => {
      const { data } = await supabase.auth.getSession();
      if (data.session) router.replace('/');
    };
    checkSession();
  }, [router]);

  const saveProfile = async (userId: string) => {
    await supabase.from('profiles').upsert({
      id: userId,
      full_name: fullName.trim(),
      username: username.trim(),
      phone: phone.trim(),
      email: email.trim(),
      updated_at: new Date().toISOString(),
    });
  };

  const signInWithGoogle = async () => {
    setError(null);
    setLoading(true);
    const origin = typeof window !== 'undefined' ? window.location.origin : '';
    const { error: authError } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: `${origin}/auth/callback` },
    });
    if (authError) {
      setError(authError.message);
      setLoading(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!fullName.trim() || !username.trim() || !email.trim() || !phone.trim() || !password) {
      setError('Please fill in name, username, email, phone, and password.');
      return;
    }

    setLoading(true);
    const { data, error: authError } = await supabase.auth.signUp({
      email: email.trim(),
      password,
      options: {
        data: {
          full_name: fullName.trim(),
          username: username.trim(),
          phone: phone.trim(),
        },
      },
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    if (data.user) {
      try {
        await saveProfile(data.user.id);
      } catch {
        // profiles table may not exist yet — metadata is still stored on the user
      }
    }

    if (data.session) {
      router.replace('/');
      return;
    }

    setSuccess('Account created! Check your email to confirm, then log in.');
    setMode('login');
    setLoading(false);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!email.trim() || !password) {
      setError('Please enter your email and password.');
      return;
    }

    setLoading(true);
    const { error: authError } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    router.replace('/');
  };

  const inputClass =
    'w-full rounded-xl border border-slate-200 bg-white/90 px-4 py-3 text-sm text-slate-900 placeholder-slate-400 transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200';

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-50 via-white to-indigo-50/40">
      <div className="pointer-events-none absolute -left-32 top-0 h-96 w-96 rounded-full bg-indigo-200/30 blur-3xl" />
      <div className="pointer-events-none absolute -right-32 bottom-0 h-96 w-96 rounded-full bg-violet-200/30 blur-3xl" />

      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-4 py-12">
        <motion.div
          className="mb-8 text-center"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Link href="/" className="inline-flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-xl text-white shadow-lg shadow-indigo-500/30">
              📦
            </div>
            <div className="text-left">
              <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">Supply Chain</p>
              <p className="text-lg font-bold text-slate-900">Price Predictor</p>
            </div>
          </Link>
        </motion.div>

        <motion.div
          className="w-full max-w-xl rounded-3xl border border-slate-200/60 bg-white/80 p-8 shadow-2xl shadow-indigo-200/20 backdrop-blur-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="mb-6 text-center">
            <h1 className="text-2xl font-bold text-slate-900">
              {mode === 'signup' ? 'Create your account' : 'Welcome back'}
            </h1>
            <p className="mt-2 text-sm text-slate-600">
              {mode === 'signup'
                ? 'First time here? Tell us a bit about you.'
                : 'Sign in with your email to continue.'}
            </p>
          </div>

          <motion.button
            type="button"
            onClick={signInWithGoogle}
            disabled={loading}
            className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800 shadow-sm transition hover:bg-slate-50 disabled:opacity-60"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
          >
            <GoogleIcon />
            Continue with Google
          </motion.button>

          <div className="my-6 flex items-center gap-3">
            <div className="h-px flex-1 bg-slate-200" />
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">or</span>
            <div className="h-px flex-1 bg-slate-200" />
          </div>

          <div className="mb-6 flex rounded-xl bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => { setMode('signup'); setError(null); setSuccess(null); }}
              className={`flex-1 rounded-lg py-2 text-sm font-semibold transition ${
                mode === 'signup' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600'
              }`}
            >
              Sign up
            </button>
            <button
              type="button"
              onClick={() => { setMode('login'); setError(null); setSuccess(null); }}
              className={`flex-1 rounded-lg py-2 text-sm font-semibold transition ${
                mode === 'login' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600'
              }`}
            >
              Log in
            </button>
          </div>

          <AnimatePresence mode="wait">
            {mode === 'signup' ? (
              <motion.form
                key="signup"
                onSubmit={handleSignup}
                className="space-y-4"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div>
                    <label className="mb-2 block text-sm font-semibold text-slate-700">Full name</label>
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="Jane Doe"
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className="mb-2 block text-sm font-semibold text-slate-700">Username</label>
                    <input
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="janedoe"
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className="mb-2 block text-sm font-semibold text-slate-700">Email</label>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@company.com"
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className="mb-2 block text-sm font-semibold text-slate-700">Phone number</label>
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      placeholder="+1 555 000 0000"
                      className={inputClass}
                    />
                  </div>
                  <div className="sm:col-span-2">
                    <label className="mb-2 block text-sm font-semibold text-slate-700">Password</label>
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••"
                      className={inputClass}
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/30 transition hover:shadow-indigo-500/50 disabled:opacity-60"
                >
                  {loading ? 'Creating account…' : 'Create account'}
                </button>
              </motion.form>
            ) : (
              <motion.form
                key="login"
                onSubmit={handleLogin}
                className="space-y-4"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
              >
                <div>
                  <label className="mb-2 block text-sm font-semibold text-slate-700">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@company.com"
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-semibold text-slate-700">Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className={inputClass}
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/30 transition hover:shadow-indigo-500/50 disabled:opacity-60"
                >
                  {loading ? 'Signing in…' : 'Log in'}
                </button>
              </motion.form>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {error && (
              <motion.p
                className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {error}
              </motion.p>
            )}
            {success && (
              <motion.p
                className="mt-4 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {success}
              </motion.p>
            )}
          </AnimatePresence>

          <p className="mt-6 text-center text-sm text-slate-500">
            <Link href="/" className="font-medium text-indigo-600 hover:text-indigo-700">
              ← Back to dashboard
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
