-- Run in Supabase SQL editor to store signup profile fields
create table if not exists public.profiles (
  id uuid references auth.users on delete cascade primary key,
  full_name text,
  username text unique,
  phone text,
  email text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table public.profiles enable row level security;

create policy "Users can view own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can insert own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);
