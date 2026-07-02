import axios from 'axios';
import type {
  StatsResponse,
  PredictRequest,
  PredictResponse,
  ExpertSuggestionRequest,
  ExpertSuggestionResponse,
  WeatherResponse,
  InsightRequest,
  PredictionHistoryItem,
  ChatRequest,
  ChatResponse,
  OnboardingRequest,
  OnboardingResponse,
} from '../types';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8001',
  headers: { 'Content-Type': 'application/json' },
  timeout: 10000,
});

export const getStats = async (): Promise<StatsResponse> => {
  const res = await api.get('/api/stats');
  return res.data;
};

export const predict = async (payload: PredictRequest): Promise<PredictResponse> => {
  const res = await api.post('/api/predict', payload);
  return res.data;
};

export const getExpertSuggestions = async (
  payload: ExpertSuggestionRequest,
): Promise<ExpertSuggestionResponse> => {
  const res = await api.post('/api/expert-suggestions', payload);
  return res.data;
};

export const getWeather = async (): Promise<WeatherResponse> => {
  const res = await api.get('/api/weather');
  return res.data;
};

export const postInsights = async (payload: InsightRequest) => {
  const res = await api.post('/api/insights', payload);
  return res.data;
};

export const getHistory = async (): Promise<PredictionHistoryItem[]> => {
  const res = await api.get('/api/predictions/history');
  return res.data.history;
};

export const deleteHistory = async (id?: string) => {
  const url = id ? `/api/predictions/history/${id}` : '/api/predictions/history';
  const res = await api.delete(url);
  return res.data;
};

export const postChat = async (payload: ChatRequest): Promise<ChatResponse> => {
  const res = await api.post('/api/chat', payload);
  return res.data;
};

export const postOnboarding = async (payload: OnboardingRequest): Promise<OnboardingResponse> => {
  const res = await api.post('/api/onboarding', payload);
  return res.data;
};

export default api;
