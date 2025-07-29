# DeepInvestor – Deep Learning-Based Portfolio Management System

**DeepInvestor** is an AI-driven portfolio management system that leverages deep learning and financial analytics to assist investors in making informed decisions. It combines stock, gold, and bond analysis into a single platform with a future roadmap for reinforcement learning integration.

---

## Features

### Stock Price Prediction
- Utilizes **Recurrent Neural Networks (RNN)** to predict stock prices for the next 3 days.
- Trained on historical stock market data fetched using the **yFinance** API.

### Beta-Based Stock Comparison
- Implements a **comparison engine** to find the top 5 companies with similar **beta scores** to the selected stock.
- Enhances portfolio diversification by suggesting similarly volatile assets.

### Gold Price Prediction
- RNN-based prediction model for **gold prices** with short-term (3-day) forecasting.
- Helps in commodity investment planning.

### Bonds Return Calculator
- Computes **expected returns** on bond investments based on amount, interest rate, and duration.
- Provides stable investment tracking for risk-averse users.

### Unified Streamlit Dashboard
- Fully integrated **Streamlit-based web interface** for interactive user experience.
- Real-time calculation of **total returns**, **profit/loss**, and portfolio summary based on user inputs.

### Future Development
- Integration of a **Reinforcement Learning model** for dynamic portfolio optimization and smart asset allocation.
- Enhanced risk management and adaptive learning from market trends.

---

## Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Programming Language:** Python
- **Web Interface:** Streamlit
- **Data Source:** yFinance API
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Others:** NumPy, Pandas, Scikit-learn


---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DeepInvestor.git
   cd DeepInvestor
