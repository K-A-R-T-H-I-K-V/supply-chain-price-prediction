"use client";
import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface ButtonProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  variant?: 'primary' | 'secondary' | 'ghost';
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({ 
  variant = 'primary', 
  className, 
  children, 
  ...rest 
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-bold rounded-xl transition px-5 py-3 text-sm';
  
  const variantStyles = {
    primary: 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 disabled:opacity-60 disabled:cursor-not-allowed',
    secondary: 'bg-slate-100 text-slate-900 hover:bg-slate-200 border border-slate-200 disabled:opacity-60 disabled:cursor-not-allowed',
    ghost: 'bg-transparent text-indigo-600 border-2 border-indigo-200 hover:bg-indigo-50 disabled:opacity-60 disabled:cursor-not-allowed',
  };

  return (
    <motion.button
      className={clsx(baseStyles, variantStyles[variant], className)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      {...(rest as any)}
    >
      {children}
    </motion.button>
  );
};

export default Button;
