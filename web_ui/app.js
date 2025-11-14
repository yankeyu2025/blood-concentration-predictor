document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('feature-form');
  const thr = document.getElementById('threshold');
  const thrVal = document.getElementById('threshold-value');
  const btnFill = document.getElementById('fill-example');
  const btnSubmit = document.getElementById('submit');
  const resultCard = document.getElementById('result-card');
  const elProb = document.getElementById('prob');
  const elThr = document.getElementById('thr');
  const elRisk = document.getElementById('risk');

  const FEATURE_NAMES = Array.isArray(window.FEATURE_NAMES) ? window.FEATURE_NAMES : [];

  const updateThr = () => {
    const v = Number(thr.value || 0.5);
    thrVal.textContent = v.toFixed(2);
  };
  thr.addEventListener('input', updateThr);
  updateThr();

  const SAMPLE = {
    'CLCR': 90,
    'height': 170,
    'Daily dose(g)': 1.5,
    'ALB(g/L)': 38,
    'ALP(U/L)': 80,
    'GGT(U/L)': 40,
    'Na(mmol/L)': 140,
  };

  btnFill.addEventListener('click', (e) => {
    e.preventDefault();
    FEATURE_NAMES.forEach((name, idx) => {
      const input = document.getElementById(`f_${idx+1}`);
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
        const input = document.getElementById(`f_${idx+1}`);
        if (!input) return;
        inst[name] = input.value;
      });

      const payload = { instances: [inst] };
      const threshold = Number(thr.value || 0.5);
      const resp = await fetch(`/predict?threshold=${threshold}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await resp.json();
      if (!data.ok) {
        elProb.textContent = '—';
        elThr.textContent = thr.value;
        const ERR = (window.TEXTS && window.TEXTS.error) || '错误';
        elRisk.textContent = ERR;
        elRisk.classList.remove('ok');
        elRisk.classList.add('bad');
        resultCard.classList.remove('hidden');
        return;
      }

      const prob = Array.isArray(data.pos_proba) ? Number(data.pos_proba[0]) : NaN;
      const thrShow = Number(data.threshold ?? threshold);
      const pred = Array.isArray(data.pred) ? Number(data.pred[0]) : (prob >= thrShow ? 1 : 0);

      elProb.textContent = isNaN(prob) ? '—' : prob.toFixed(4);
      elThr.textContent = thrShow.toFixed(2);
      const RISK_BAD = (window.TEXTS && window.TEXTS.risk_insufficient) || '低血药浓度风险';
      const RISK_OK  = (window.TEXTS && window.TEXTS.risk_adequate) || '血药浓度达标或偏高';
      elRisk.textContent = pred === 0 ? RISK_BAD : RISK_OK;
      elRisk.classList.remove('ok', 'bad');
      elRisk.classList.add(pred === 0 ? 'bad' : 'ok');

      resultCard.classList.remove('hidden');
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
      elProb.textContent = '—';
      elThr.textContent = thr.value;
      elRisk.textContent = (window.TEXTS && window.TEXTS.error) || '错误';
      elRisk.classList.remove('ok');
      elRisk.classList.add('bad');
      resultCard.classList.remove('hidden');
      console.error('预测失败：', err);
    }
  });
});
