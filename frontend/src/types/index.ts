export interface StatsResponse {
  total_predictions: number;
  avg_price: number;
  active_models: string[];
  last_prediction?: { id: string; price: number; when: string } | null;
}

export interface PredictRequest {
  order_quantity: number;
  discount: number;
  shipping_cost: number;
  product_base_margin: number;
  product_category: string;
  month: number;
  ship_mode: string;
  order_priority: string;
}

export interface PredictResponse {
  predicted_unit_price: number;
  confidence?: number;
}

export interface ExpertSuggestionRequest extends PredictRequest {
  predicted_price: number;
}

export interface ExpertSuggestionResponse {
  message: string;
  suggestions: string[];
  summary?: string;
}

export interface WeatherResponse {
  summary: string;
  temperature?: number;
}

export interface InsightRequest {
  timeframe?: string;
  metrics?: string[];
}

export interface PredictionHistoryItem {
  timestamp: string;
  request: PredictRequest;
  predicted_unit_price: number;
}

export interface ChatRequest {
  message: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  reply: string;
}

