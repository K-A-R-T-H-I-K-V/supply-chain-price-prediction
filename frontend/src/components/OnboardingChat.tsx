"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { postOnboarding } from '@/services/api';

const ONBOARDING_KEY = 'scpp_onboarding_complete';

type ChatAction = {
  label: string;
  value: string;
  variant?: 'primary' | 'outline';
};

type ChatMessage = {
  id: string;
  role: 'bot' | 'user';
  text: string;
  actions?: ChatAction[];
  answered?: boolean;
};

type Phase = 'greeting' | 'explain' | 'invite' | 'done';

type Props = {
  onComplete: () => void;
};

const GREETING_MESSAGE =
  "Hi there! 👋 Before we dive in — do you know about supply chain?";

const TypingIndicator = () => (
  <div className="flex items-center gap-1 px-4 py-3">
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        className="h-2 w-2 rounded-full bg-indigo-400"
        animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
        transition={{ repeat: Infinity, duration: 1, delay: i * 0.15 }}
      />
    ))}
  </div>
);

const ActionButtons = ({
  actions,
  onSelect,
  disabled,
}: {
  actions: ChatAction[];
  onSelect: (value: string) => void;
  disabled?: boolean;
}) => (
  <div className="mt-3 flex flex-wrap gap-2">
    {actions.map((action) => (
      <motion.button
        key={action.label}
        type="button"
        disabled={disabled}
        onClick={() => onSelect(action.value)}
        className={`rounded-full px-4 py-2 text-sm font-semibold transition disabled:opacity-50 ${
          action.variant === 'primary'
            ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-500/20'
            : 'border border-indigo-200 bg-white text-indigo-600 hover:bg-indigo-50'
        }`}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
      >
        {action.label}
      </motion.button>
    ))}
  </div>
);

export const OnboardingChat: React.FC<Props> = ({ onComplete }) => {
  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: 'greeting',
      role: 'bot',
      text: GREETING_MESSAGE,
      actions: [
        { label: 'Yes', value: 'yes', variant: 'outline' },
        { label: 'No', value: 'no', variant: 'primary' },
      ],
    },
  ]);
  const [phase, setPhase] = useState<Phase>('greeting');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const markAnswered = useCallback((messageId: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, answered: true, actions: undefined } : m)),
    );
  }, []);

  const addBotMessage = useCallback((text: string, actions?: ChatAction[]) => {
    setMessages((prev) => [
      ...prev,
      { id: `bot-${Date.now()}-${prev.length}`, role: 'bot', text, actions },
    ]);
  }, []);

  const addUserMessage = useCallback((text: string) => {
    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}-${prev.length}`, role: 'user', text },
    ]);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (typeof window !== 'undefined' && localStorage.getItem(ONBOARDING_KEY) === 'true') {
      onCompleteRef.current();
    }
  }, []);

  const handleKnowledgeAnswer = async (messageId: string, knows: boolean) => {
    markAnswered(messageId);
    addUserMessage(knows ? 'Yes' : 'No');
    setPhase('explain');
    setLoading(true);

    try {
      const res = await postOnboarding({ step: 'explain', knows_supply_chain: knows });
      addBotMessage(res.reply);

      const invite = await postOnboarding({ step: 'invite' });
      addBotMessage(invite.reply, [
        { label: "Yes, let's go!", value: 'start-form', variant: 'primary' },
      ]);
      setPhase('invite');
    } catch {
      addBotMessage(
        knows
          ? 'Good to know! We focus on office supplies — a small set of essentials like printer paper, pens, and staplers. Fill in the form to get a price estimate, and we factor in live weather, news, and fuel costs so the tips match real conditions.'
          : 'Supply chain is simply getting products from a supplier to you on time. We handle office supplies only — about 2–3 essentials like paper, pens, and staplers. The form gives you a price estimate, informed by live weather, news, and shipping costs.',
      );
      addBotMessage('Ready to try it? Fill in the prediction form with your order details.', [
        { label: "Yes, let's go!", value: 'start-form', variant: 'primary' },
      ]);
      setPhase('invite');
    } finally {
      setLoading(false);
    }
  };

  const handleStartForm = (messageId: string) => {
    markAnswered(messageId);
    addUserMessage("Yes, let's go!");
    setPhase('done');
    localStorage.setItem(ONBOARDING_KEY, 'true');
    onComplete();
  };

  const handleAction = (messageId: string, value: string) => {
    if (phase === 'greeting') {
      handleKnowledgeAnswer(messageId, value === 'yes');
      return;
    }
    if (phase === 'invite' && value === 'start-form') {
      handleStartForm(messageId);
    }
  };

  return (
    <div className="flex h-full min-h-[520px] max-h-[calc(100vh-180px)] flex-col rounded-2xl border border-slate-200/50 bg-white shadow-xl shadow-slate-200/30 overflow-hidden">
      <div className="shrink-0 border-b border-slate-200/50 px-6 py-4">
        <h2 className="text-lg font-semibold text-slate-900">Quick intro</h2>
        <p className="text-sm text-slate-500">A short overview before you predict</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">
        <div className="mx-auto max-w-2xl space-y-5">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[90%] ${msg.role === 'user' ? 'text-right' : ''}`}>
                  <div
                    className={`inline-block rounded-2xl px-4 py-3 text-left text-sm leading-6 ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-500/20'
                        : 'border border-slate-200/60 bg-slate-50/90 text-slate-800'
                    }`}
                  >
                    {msg.text}
                  </div>

                  {msg.role === 'bot' && msg.actions && !msg.answered && (
                    <ActionButtons
                      actions={msg.actions}
                      onSelect={(value) => handleAction(msg.id, value)}
                      disabled={loading}
                    />
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-slate-200/60 bg-slate-50/80">
                <TypingIndicator />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
};

export default OnboardingChat;
