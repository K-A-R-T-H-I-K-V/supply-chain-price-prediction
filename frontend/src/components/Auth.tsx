"use client";
import React, { useEffect, useState } from 'react';
import supabase from '@/lib/supabaseClient';

export const Auth: React.FC = () => {
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    const get = async () => {
      const { data } = await supabase.auth.getSession();
      setUser(data?.session?.user ?? null);
    };
    get();

    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    const subscription = (data as any)?.subscription;
    return () => subscription?.unsubscribe?.();
  }, []);

  const signInWithGoogle = async () => {
    await supabase.auth.signInWithOAuth({ provider: 'google' });
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
  };

  if (user) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-700">{user.email}</span>
        <button onClick={signOut} className="rounded px-3 py-1 text-sm bg-slate-100">Sign out</button>
      </div>
    );
  }

  return (
    <div>
      <button onClick={signInWithGoogle} className="rounded bg-indigo-600 px-3 py-2 text-sm text-white">Sign in with Google</button>
    </div>
  );
};

export default Auth;
