# app.py

import os
import sys
import uuid
import importlib.util
import io
import base64

from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from portwine.backtester import Backtester
from portwine.loaders.eodhd import EODHDMarketDataLoader
from portwine.analyzers.equitydrawdown import EquityDrawdownAnalyzer
from portwine.strategies.base import StrategyBase

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/portwine_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = '''
<!doctype html>
<html>
  <head>
    <title>Portwine MVP</title>
  </head>
  <body>
    <h1>Paste Your StrategyBase Subclass</h1>
    <form method="post">
      <textarea name="code" rows="20" cols="80"
        placeholder="Paste your StrategyBase subclass here..."></textarea><br>
      <label for="tickers">Tickers (comma-separated):</label><br>
      <input type="text" name="tickers" size="80"
             placeholder="e.g. AAPL, MSFT, GOOG"/><br><br>
      <button type="submit">Run Backtest</button>
    </form>
    {% if img_data %}
      <h2>Equity Curve & Drawdown</h2>
      <img src="data:image/png;base64,{{img_data}}" alt="Equity & Drawdown Plot"/>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None

    if request.method == 'POST':
        raw_code = request.form.get('code', '')
        tickers_input = request.form.get('tickers', '')
        if not raw_code.strip():
            return 'Error: No strategy code provided.', 400
        if not tickers_input.strip():
            return 'Error: No tickers provided.', 400

        # Parse tickers list
        tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        if not tickers:
            return 'Error: Could not parse any tickers.', 400

        # Prepend import so StrategyBase is defined
        full_code = "from portwine.strategies.base import StrategyBase\n\n" + raw_code

        # Write user code to a temporary file
        filename = f"{uuid.uuid4().hex}.py"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'w') as f:
            f.write(full_code)

        # Dynamically import the module
        spec = importlib.util.spec_from_file_location('user_strategy', filepath)
        user_mod = importlib.util.module_from_spec(spec)
        user_mod.__dict__['StrategyBase'] = StrategyBase
        sys.modules['user_strategy'] = user_mod
        try:
            spec.loader.exec_module(user_mod)
        except Exception as e:
            return f'Error importing strategy code: {e}', 400

        # Find the first subclass of StrategyBase
        strategy_cls = None
        for obj in vars(user_mod).values():
            if isinstance(obj, type) and issubclass(obj, StrategyBase) and obj is not StrategyBase:
                strategy_cls = obj
                break
        if strategy_cls is None:
            return 'Error: No StrategyBase subclass found in code.', 400

        # Instantiate and run backtest
        try:
            strategy = strategy_cls(tickers)
            data_loader = EODHDMarketDataLoader(data_path='/Users/stuart/Developer/Data/EODHD/us_sorted/US')
            backtester = Backtester(data_loader)
            results = backtester.run_backtest(strategy)
            if results is None:
                return ('Error: Backtest returned no results. '
                        'Make sure your CSV files live under ./data with names like TICKER.US.csv, '
                        'and that your tickers are valid.'), 500
        except Exception as e:
            return f'Error running backtest: {e}', 500

        # Plot equity & drawdown
        try:
            eq_analyzer = EquityDrawdownAnalyzer()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            strat_eq = (1.0 + results['strategy_returns']).cumprod()
            bm_eq    = (1.0 + results['benchmark_returns']).cumprod()

            ax1.plot(strat_eq.index, strat_eq.values, label='Strategy')
            ax1.plot(bm_eq.index,    bm_eq.values,    label='Benchmark')
            ax1.set_title('Equity Curve')
            ax1.legend()
            ax1.grid(True)

            strat_dd = eq_analyzer.compute_drawdown(strat_eq) * 100
            bm_dd    = eq_analyzer.compute_drawdown(bm_eq)    * 100

            ax2.plot(strat_dd.index, strat_dd.values, label='Strategy DD (%)')
            ax2.plot(bm_dd.index,    bm_dd.values,    label='Benchmark DD (%)')
            ax2.set_title('Drawdown (%)')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('ascii')
            plt.close(fig)
        except Exception as e:
            return f'Error generating plot: {e}', 500

    return render_template_string(HTML_TEMPLATE, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
