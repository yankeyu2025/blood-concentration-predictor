document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('feature-form');
  const FIXED_THR = 0.5;
  const btnFill = document.getElementById('fill-example');
  const btnSubmit = document.getElementById('submit');
  const resultCard = document.getElementById('result-card');
  const explainCard = document.getElementById('explain-card');
  const elRisk = document.getElementById('risk');
  const elWaterfall = document.getElementById('waterfall');

  const FEATURE_NAMES = Array.isArray(window.FEATURE_NAMES) ? window.FEATURE_NAMES : [];

  const SAMPLE = {
    'height': 170,
    'weight': 65,
    'Daily dose(g)': 0.6,
    'ALB(g/L)': 45,
    'ALP(U/L)': 90
  };

  function toDisplayLabel(name) {
    return (window.TEXTS && window.TEXTS.feature_units && window.TEXTS.feature_units[name]) || name;
  }

  function showErrorState(message) {
    const fallback = (window.TEXTS && window.TEXTS.error) || 'Error';
    elRisk.textContent = message || fallback;
    elRisk.classList.remove('ok');
    elRisk.classList.add('bad');
    resultCard.classList.remove('hidden');
    if (explainCard) explainCard.classList.add('hidden');
  }

  function renderWaterfall(shapData) {
    if (!elWaterfall) return;
    elWaterfall.innerHTML = '';
    if (!shapData || !Array.isArray(shapData.features) || shapData.features.length === 0) {
      elWaterfall.classList.add('hidden');
      return;
    }

    const features = shapData.features.slice(0, Number(shapData.top_n || 5));
    const maxAbs = Math.max(...features.map((f) => Math.abs(Number(f.shap) || 0)), 1e-6);

    features.forEach((item) => {
      const shapVal = Number(item.shap) || 0;
      const value = Number(item.value);
      const row = document.createElement('div');
      row.className = 'wf-row';

      const meta = document.createElement('div');
      meta.className = 'wf-meta';
      meta.textContent = `${toDisplayLabel(item.feature)} = ${Number.isFinite(value) ? value.toFixed(2) : item.value}`;

      const track = document.createElement('div');
      track.className = 'wf-track';

      const zero = document.createElement('div');
      zero.className = 'wf-zero';
      track.appendChild(zero);

      const bar = document.createElement('div');
      bar.className = `wf-bar ${shapVal >= 0 ? 'pos' : 'neg'}`;
      const widthPct = (Math.abs(shapVal) / maxAbs) * 48;
      bar.style.width = `${Math.max(widthPct, 2)}%`;
      if (shapVal >= 0) {
        bar.style.left = '50%';
      } else {
        bar.style.left = `${50 - Math.max(widthPct, 2)}%`;
      }
      track.appendChild(bar);

      const val = document.createElement('div');
      val.className = `wf-value ${shapVal >= 0 ? 'pos' : 'neg'}`;
      val.textContent = `${shapVal >= 0 ? '+' : ''}${shapVal.toFixed(4)}`;

      row.appendChild(meta);
      row.appendChild(track);
      row.appendChild(val);
      elWaterfall.appendChild(row);
    });

    elWaterfall.classList.remove('hidden');
  }

  btnFill.addEventListener('click', (e) => {
    e.preventDefault();
    FEATURE_NAMES.forEach((name, idx) => {
      const input = document.getElementById(`f_${idx + 1}`);
      if (!input) return;
      if (Object.prototype.hasOwnProperty.call(SAMPLE, name)) {
        input.value = SAMPLE[name];
      }
    });
  });

  btnSubmit.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
      const inst = {};
      FEATURE_NAMES.forEach((name, idx) => {
        const input = document.getElementById(`f_${idx + 1}`);
        if (!input) return;
        inst[name] = input.value;
      });

      const payload = { instances: [inst] };
      const threshold = FIXED_THR;
      const resp = await fetch(`/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      let data = null;
      try {
        data = await resp.json();
      } catch (parseError) {
        throw new Error(`invalid response payload (${resp.status})`);
      }

      if (!resp.ok) {
        throw new Error((data && data.error) || `request failed (${resp.status})`);
      }

      if (!data.ok) {
        const missing = Array.isArray(data.missing) ? data.missing : [];
        const missingLabel = (window.TEXTS && window.TEXTS.missing_fields_prefix) || 'Missing fields';
        const errorMsg = missing.length > 0
          ? `${missingLabel}: ${missing.map(toDisplayLabel).join(', ')}`
          : (data.error || (window.TEXTS && window.TEXTS.error) || 'Error');
        showErrorState(errorMsg);
        return;
      }

      const prob = Array.isArray(data.pos_proba) ? Number(data.pos_proba[0]) : NaN;
      const thrShow = Number(data.threshold ?? threshold);
      const pred = Array.isArray(data.pred) ? Number(data.pred[0]) : (prob >= thrShow ? 1 : 0);

      const RISK_BAD = (window.TEXTS && window.TEXTS.risk_insufficient) || '低血药浓度风险';
      const RISK_OK = (window.TEXTS && window.TEXTS.risk_adequate) || '血药浓度达标或偏高';
      elRisk.textContent = pred === 0 ? RISK_BAD : RISK_OK;
      elRisk.classList.remove('ok', 'bad');
      elRisk.classList.add(pred === 0 ? 'bad' : 'ok');

      renderWaterfall(data.shap);
      if (explainCard) explainCard.classList.remove('hidden');

      resultCard.classList.remove('hidden');
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
      showErrorState(err && err.message ? err.message : ((window.TEXTS && window.TEXTS.error) || 'Error'));
      console.error('预测失败：', err);
    }
  });
});
